import argparse
import time

import numpy as np
import tqdm
import ray
import wandb
import torch
import torch.nn.functional as F

from .rollout import RayRollout, Reward
from .replay_buffer import ReplayBuffer
from . import controller, policy


N_WORKERS = 1


class RaySampler(object):
    def __init__(self):
        self.rollouts = [RayRollout.remote() for _ in range(N_WORKERS)]

    def get_samples(self, agent, samples=2000, max_step=500):
        [ray.get(rollout.start.remote()) for rollout in self.rollouts]

        tick = time.time()

        ros = list()
        total = 0

        while total < samples:
            batch_ros = list()

            for rollout in self.rollouts:
                batch_ros.append(rollout.rollout.remote(
                        agent, controller.TuxController(),
                        max_step=max_step, restart=True))

            batch_ros = [ray.get(ro) for ro in batch_ros]

            ros.extend(batch_ros)
            total += sum(map(len, batch_ros))

        clock = time.time() - tick

        print('FPS: %.2f' % (sum(map(len, ros)) / clock))
        print('AFPS: %.2f' % (np.mean(list(map(len, ros))) / clock))

        [ray.get(rollout.stop.remote()) for rollout in self.rollouts]

        wandb.log({
            'fps': (sum(map(len, ros)) / clock),
            'afps': (np.mean(list(map(len, ros))) / clock),
            },
            step=wandb.run.summary['step'])

        return ros


def log_video(rollouts):
    videos = list()

    for rollout in rollouts:
        video = list()

        for data in rollout:
            video.append(data.s.transpose(2, 0, 1))

        videos.append(np.uint8(video))

    t, c, h, w = videos[0].shape

    full = np.zeros((t, c, h * 2, w * 2), dtype=np.uint8)
    full[:, :, :h, :w] = videos[0]
    full[:, :, h:, :w] = videos[1]
    full[:, :, :h, w:] = videos[2]
    full[:, :, h:, w:] = videos[3]

    wandb.log({
        'video': [wandb.Video(full, format='mp4', fps=20)]},
        step=wandb.run.summary['step'])


def train(net, replay):
    s, a, sp, g, done = replay[[1, 0]]

    s = torch.FloatTensor(s.transpose(0, 3, 1, 2))
    s = s.to(config['device'])

    a_hat = net(s)


def main(config):
    wandb.init(project='rl', config=config)
    wandb.run.summary['step'] = 0

    replay = ReplayBuffer()
    net = policy.DeepNet()
    net.to(config['device'])

    for epoch in tqdm.tqdm(range(config['max_epoch']+1), desc='epoch'):
        wandb.run.summary['epoch'] = epoch
        wandb.run.summary['step'] += 1

        rollouts = RaySampler().get_samples(policy.DeepPolicy(net))

        for rollout in rollouts:
            for data in rollout:
                replay.add(data)

        log_video(rollouts[:4])

        train(net, replay)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=1)
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
