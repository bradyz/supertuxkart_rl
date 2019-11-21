import argparse
import time

import numpy as np
import ray
import wandb
import torch

from .utils import make_video
from .rollout import RayRollout
from .replay_buffer import ReplayBuffer
from .reinforce import REINFORCE
from .ppo import PPO
from .ddpg import DDPG


N_WORKERS = 3


class RaySampler(object):
    def __init__(self):
        random_track = lambda: np.random.choice(["lighthouse", "zengarden", "hacienda", "snowtuxpeak", "cornfield_crossing"])
        random_track = lambda: np.random.choice(["lighthouse"])
        self.rollouts = [RayRollout.remote(random_track()) for _ in range(N_WORKERS)]

        # tracks = ["lighthouse", "zengarden", "hacienda", "snowtuxpeak", "cornfield_crossing"]
        # self.rollouts = [RayRollout.remote(track) for track in tracks]

    def get_samples(self, agent, max_frames=10000, max_step=500, gamma=1.0, frame_skip=0, **kwargs):
        tick = time.time()
        total_frames = 0
        returns = list()
        video_rollouts = list()

        while total_frames <= max_frames:
            batch_ros = list()

            for rollout in self.rollouts:
                batch_ros.append(
                        rollout.rollout.remote(
                            agent,
                            max_step=max_step, gamma=gamma, frame_skip=frame_skip))

            batch_ros = ray.get(batch_ros)
            # batch_ros = [ray.get(ro) for ro in batch_ros]

            if len(video_rollouts) < 64:
                video_rollouts.extend([ro for ro, ret in batch_ros if len(ro) > 0])

            total_frames += sum(len(ro) * (frame_skip + 1) for ro, ret in batch_ros)
            returns.extend([ret for ro, ret in batch_ros])

            yield batch_ros

        clock = time.time() - tick

        print('FPS: %.2f' % (total_frames / clock))
        print('Count: %d' % (total_frames))
        print('Episodes: %d' % (len(returns)))
        print('Time: %.2f' % clock)
        print('Return: %.3f' % np.mean(returns))

        wandb.run.summary['frames'] = wandb.run.summary.get('frames', 0) + total_frames
        wandb.run.summary['episodes'] = wandb.run.summary.get('episodes', 0) + len(returns)

        wandb.log({
            'video': [wandb.Video(make_video(video_rollouts), format='mp4', fps=20)],

            'epoch/fps': (total_frames / clock),
            'epoch/episodes': len(returns),
            'epoch/return': np.mean(returns),

            'total/frames': wandb.run.summary['frames'],
            'total/episodes': wandb.run.summary['episodes'],
            }, step=wandb.run.summary['step'])


def main(config):
    wandb.init(project='test', config=config)
    wandb.run.summary['step'] = 0

    trainer = {
            'reinforce': REINFORCE,
            'ppo': PPO,
            'ddpg': DDPG,
            }[config['algorithm']](**config)

    sampler = RaySampler()

    for epoch in range(config['max_epoch']+1):
        wandb.run.summary['epoch'] = epoch

        replay = ReplayBuffer(config['max_frames'])

        for rollout_batch in sampler.get_samples(trainer.get_policy(epoch), **config):
            for rollout, _ in rollout_batch:
                for data in rollout:
                    replay.add(data)

        print([x.r[0] for x in rollout])

        metrics = trainer.train(replay)

        wandb.log(metrics, step=wandb.run.summary['step'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=10000)

    # Optimizer args.
    parser.add_argument('--algorithm', type=str, default='reinforce')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_1', type=float, default=1e-4)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--max_frames', type=int, default=5000)
    parser.add_argument('--frame_skip', type=int, default=0)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--clip', type=float, default=0.1)
    parser.add_argument('--importance_sampling', action='store_true', default=False)

    parsed = parser.parse_args()

    config = {
            'algorithm': parsed.algorithm,
            'frame_skip': parsed.frame_skip,
            'max_frames': parsed.max_frames,
            'gamma': parsed.gamma,

            'max_epoch': parsed.max_epoch,
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

            'batch_size': parsed.batch_size,
            'iterations': parsed.iterations,
            'lr': parsed.lr,
            'lr_1': parsed.lr_1,
            'clip': parsed.clip,
            'tau': parsed.tau,
            'eps': parsed.eps,
            'importance_sampling': parsed.importance_sampling,
            }

    ray.init(logging_level=40, num_cpus=N_WORKERS, num_gpus=1)

    main(config)
