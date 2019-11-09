import argparse
import time

import numpy as np
import ray
import wandb
import torch

from . import controller, utils
from .rollout import RayRollout, Rollout
from .replay_buffer import ReplayBuffer
from .reinforce import REINFORCE
from .ppo import PPO
from .ddpg import DDPG


N_WORKERS = 4


class RaySampler(object):
    def __init__(self):
        self.rollouts = [RayRollout.remote() for _ in range(N_WORKERS)]

    def get_samples(self, agent, max_step=2000, gamma=1.0):
        random_track = lambda: np.random.choice(["lighthouse", "zengarden", "hacienda", "sandtrack", "volcano_island"])
        random_track = lambda: np.random.choice(["lighthouse"])

        [ray.get(rollout.start.remote(track=random_track())) for rollout in self.rollouts]

        tick = time.time()
        total = 0
        total_episodes = 0

        while total <= 10000:
            batch_ros = list()

            for rollout in self.rollouts:
                batch_ros.append(rollout.rollout.remote(
                        agent, controller.TuxController(),
                        max_step=max_step, restart=True, gamma=gamma))

            batch_ros = [ray.get(ro) for ro in batch_ros]
            total += sum(len(x[0]) for x in batch_ros)
            total_episodes += len(batch_ros)

            yield batch_ros

        clock = time.time() - tick

        print('FPS: %.2f' % (total / clock))
        print('Count: %d' % (total))
        print('Episodes: %d' % (total_episodes))
        print('Time: %.2f' % clock)

        [ray.get(rollout.stop.remote()) for rollout in self.rollouts]

        wandb.run.summary['frames'] = wandb.run.summary.get('frames', 0) + total

        wandb.log({
            'fps': (total / clock),
            'episodes': total_episodes,
            'frames': wandb.run.summary['frames'],
            }, step=wandb.run.summary['step'])


def main(config):
    wandb.init(project='test', config=config)
    wandb.run.summary['step'] = 0

    replay = ReplayBuffer(max_size=10000)

    rollout = Rollout()
    rollout.start()

    if config['algorithm'] == 'reinforce':
        trainer = REINFORCE(**config)
    elif config['algorithm'] == 'ppo':
        trainer = PPO(**config)
    elif config['algorithm'] == 'ddpg':
        trainer = DDPG(**config)

    for epoch in range(config['max_epoch']+1):
        wandb.run.summary['epoch'] = epoch

        returns = list()
        video_rollouts = list()

        for rollout_batch in RaySampler().get_samples(trainer.get_policy(), gamma=config['gamma']):
            for rollout, r_total in rollout_batch:
                if len(video_rollouts) < 64:
                    video_rollouts.append(rollout)

                returns.append(r_total)

                for data in rollout:
                    replay.add(data)

        wandb.log({
            'return': np.mean(returns),
            'video': [wandb.Video(utils.make_video(video_rollouts), format='mp4', fps=20)]
            },
            step=wandb.run.summary['step'])

        metrics = trainer.train(replay)

        wandb.log(metrics, step=wandb.run.summary['step'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=10000)

    # Optimizer args.
    parser.add_argument('--algorithm', type=str, default='reinforce')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--importance_sampling', action='store_true', default=False)

    parsed = parser.parse_args()

    config = {
            'algorithm': parsed.algorithm,
            'max_epoch': parsed.max_epoch,

            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

            'batch_size': parsed.batch_size,
            'lr': parsed.lr,
            'gamma': parsed.gamma,
            'eps': parsed.eps,
            'importance_sampling': parsed.importance_sampling,
            }

    ray.init(logging_level=40, num_cpus=N_WORKERS, num_gpus=1)

    main(config)
