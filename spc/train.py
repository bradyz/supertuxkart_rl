import argparse
import time

import numpy as np
import torch
import tqdm
import ray
import wandb

from .rollout import RayRollout, Reward
from .replay_buffer import ReplayBuffer
from . import controller, policy


N_WORKERS = 1


class RaySampler(object):
    def __init__(self):
        self.rollouts = [RayRollout.remote() for _ in range(N_WORKERS)]

    def get_samples(self, agent, max_step=1000):
        tick = time.time()

        [ray.get(rollout.start.remote()) for rollout in self.rollouts]

        ros = list()

        for rollout in self.rollouts:
            ros.append(rollout.rollout.remote(
                    agent, controller.TuxController(),
                    max_step=max_step, restart=True))

        ros = [ray.get(ro) for ro in ros]

        [ray.get(rollout.stop.remote()) for rollout in self.rollouts]

        clock = time.time() - tick

        print('FPS: %.2f' % (sum(map(len, ros)) / clock))
        print('AFPS: %.2f' % (np.mean(list(map(len, ros))) / clock))

        return ros


def log_video(rollout):
    video = list()

    for data in rollout:
        video.append(data.s.transpose(2, 0, 1))

    wandb.log({'video': [wandb.Video(np.uint8(video), format='mp4', fps=20)]})


def main(config):
    wandb.init(project='rl', config=config)
    wandb.run.summary['step'] = 0

    replay = ReplayBuffer()
    agent = policy.RewardPolicy(Reward())

    for epoch in tqdm.tqdm(range(config['max_epoch']+1), desc='epoch', position=0):
        wandb.run.summary['epoch'] = epoch

        for i, rollout in enumerate(RaySampler().get_samples(agent)):
            for data in rollout:
                replay.add(data)

            if i == 0:
                log_video(rollout)

    s, a, sp, r, done = replay[[1, 0]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)

    # Optimizer args.
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)

    parsed = parser.parse_args()

    config = {
            'max_epoch': parsed.max_epoch,
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'optimizer_args': {
                'lr': parsed.lr,
                'weight_decay': parsed.weight_decay,
                }
            }

    ray.init(logging_level=40)

    main(config)
