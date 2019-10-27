import argparse
import time

import numpy as np
import tqdm
import ray
import wandb
import torch

from .rollout import RayRollout
from .replay_buffer import ReplayBuffer
from . import controller, policy
from . import utils


N_WORKERS = 4


class RaySampler(object):
    def __init__(self):
        self.rollouts = [RayRollout.remote() for _ in range(N_WORKERS)]

    def get_samples(self, agent, iterations=20, max_step=1000, gamma=1.0):
        random_track = lambda: np.random.choice(["lighthouse", "zengarden", "hacienda", "sandtrack", "volcano_island"])
        [ray.get(rollout.start.remote(track=random_track())) for rollout in self.rollouts]

        tick = time.time()

        total = 0

        for _ in range(iterations):
            batch_ros = list()

            for rollout in self.rollouts:
                batch_ros.append(rollout.rollout.remote(
                        agent, controller.TuxController(),
                        max_step=max_step, restart=True, gamma=gamma))

            batch_ros = [ray.get(ro) for ro in batch_ros]

            total += sum(len(x[0]) for x in batch_ros)

            yield batch_ros

        clock = time.time() - tick

        print('FPS: %.2f' % (total / clock))
        print('Count: %d' % (total))
        print('Time: %.2f' % clock)

        [ray.get(rollout.stop.remote()) for rollout in self.rollouts]

        wandb.log({'fps': (total / clock)}, step=wandb.run.summary['step'])


def train(net, optim, replay, config):
    net.to(config['device'])
    net.train()

    losses = list()

    for i in range(1000):
        indices = np.random.choice(len(replay), config['batch_size'])
        s, a, g, done = replay[indices]

        s = torch.FloatTensor(s.transpose(0, 3, 1, 2))
        s = s.to(config['device'])

        a = torch.FloatTensor(a).squeeze()
        a = a.to(config['device'])

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

        returns = list()
        rollouts = list()

        for rollout_batch in RaySampler().get_samples(policy.DeepPolicy(net), gamma=config['gamma']):
            for rollout, r_total in rollout_batch:
                if len(rollouts) < 16:
                    rollouts.append(rollout)

                returns.append(r_total)

                for data in rollout:
                    replay.add(data)

        wandb.log({'return': np.mean(returns)}, step=wandb.run.summary['step'])

        wandb.log({
            'video': [wandb.Video(utils.make_video(rollouts), format='mp4', fps=20)]},
            step=wandb.run.summary['step'])

        loss_epoch = train(net, optim, replay, config)

        wandb.log({'loss': loss_epoch, 'return': np.mean(returns)}, step=wandb.run.summary['step'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=256)

    # Optimizer args.
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--gamma', type=float, default=0.5)

    parsed = parser.parse_args()

    config = {
            'max_epoch': parsed.max_epoch,
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'batch_size': parsed.batch_size,
            'lr': parsed.lr,
            'gamma': parsed.gamma,
            }

    ray.init(logging_level=40, num_cpus=N_WORKERS, num_gpus=1)

    main(config)
