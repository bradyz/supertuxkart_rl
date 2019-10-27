import argparse
import time

import numpy as np
import tqdm
import ray
import wandb
import torch

from .rollout import Rollout, RayRollout, Reward
from .replay_buffer import ReplayBuffer
from . import controller, policy


N_WORKERS = 4


class RaySampler(object):
    def __init__(self):
        self.rollouts = [RayRollout.remote() for _ in range(N_WORKERS)]

    def get_samples(self, agent, samples=80000, max_step=2000):
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
        print('Count: %d' % (sum(map(len, ros))))
        print('Time: %.2f' % clock)

        [ray.get(rollout.stop.remote()) for rollout in self.rollouts]

        wandb.log({'fps': (sum(map(len, ros)) / clock)}, step=wandb.run.summary['step'])

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


def train(net, optim, replay, config):
    net.to(config['device'])
    net.train()

    losses = list()

    for _ in range(1000):
        indices = np.random.choice(len(replay), config['batch_size'])
        s, a, sp, g, done = replay[indices]

        s = torch.FloatTensor(s.transpose(0, 3, 1, 2))
        s = s.to(config['device'])

        a = torch.FloatTensor(a).squeeze()
        a = a.to(config['device'])

        # mask = torch.FloatTensor(np.eye(8)[a.squeeze()])
        # mask = mask.to(config['device'])

        g = torch.FloatTensor(g)
        g = g.to(config['device'])
        g = (g - g.mean()) / (g.std() + 1e-7)

        m = torch.distributions.Categorical(logits=net(s))

        loss = -(g * m.log_prob(a)).sum(1)
        loss_mean = loss.mean()

        loss_mean.backward()
        optim.step()
        optim.zero_grad()

        wandb.run.summary['step'] += 1

        losses.append(loss_mean.item())

        wandb.log({'loss_batch': loss_mean.item()}, step=wandb.run.summary['step'])

    return np.mean(losses)


def main(config):
    wandb.init(project='rl', config=config)
    wandb.run.summary['step'] = 0

    net = policy.DeepNet()
    net.to(config['device'])
    optim = torch.optim.Adam(net.parameters(), lr=config['lr'])

    replay = ReplayBuffer()

    # rollout = Rollout()
    # rollout.start()

    # episode = rollout.rollout(
            # policy.DeepPolicy(net),
            # controller.TuxController(),
            # max_step=10000)

    for epoch in tqdm.tqdm(range(config['max_epoch']+1), desc='epoch'):
        wandb.run.summary['epoch'] = epoch

        rollouts = RaySampler().get_samples(policy.DeepPolicy(net), 80000)
        returns = list()

        for rollout in rollouts:
            returns.append(rollout[0].r)

            for data in rollout:
                replay.add(data)

        wandb.log({'return': np.mean(returns)}, step=wandb.run.summary['step'])

        log_video(rollouts[:4])
        loss_epoch = train(net, optim, replay, config)

        wandb.log({'loss': loss_epoch, 'return': np.mean(returns)}, step=wandb.run.summary['step'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=256)

    # Optimizer args.
    parser.add_argument('--lr', type=float, default=1e-2)

    parsed = parser.parse_args()

    config = {
            'max_epoch': parsed.max_epoch,
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'batch_size': parsed.batch_size,
            'lr': parsed.lr,
            }

    ray.init(logging_level=40, num_cpus=N_WORKERS, num_gpus=1)

    main(config)
