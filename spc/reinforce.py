import wandb
import numpy as np
import torch

from . import policy


class REINFORCE(object):
    def __init__(self, batch_size, lr, gamma, importance_sampling, device, **kwargs):
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.importance_sampling = importance_sampling

        self.device = device

        self.net = policy.DeepNet()
        self.net.to(device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def train(self, replay, n_iterations=1000):
        self.net.to(self.device)
        self.net.train()

        losses = list()

        for i in range(n_iterations):
            indices = np.random.choice(len(replay), self.batch_size)
            s, a, p_old, g, done = replay[indices]

            s = torch.FloatTensor(s.transpose(0, 3, 1, 2))
            s = s.to(self.device)

            a = torch.FloatTensor(a).squeeze()
            a = a.to(self.device)

            g = torch.FloatTensor(g)
            g = g.to(self.device)
            g = (g - g.mean()) / (g.std() + 1e-7)

            p_old = torch.FloatTensor(p_old)
            p_old = p_old.to(self.device)

            m = torch.distributions.Categorical(logits=self.net(s))
            log_p = m.log_prob(a)

            if self.importance_sampling:
                rho = torch.exp(log_p) / p_old
            else:
                rho = 1.0

            loss = -(rho * g * log_p).sum(1)
            loss_mean = loss.mean()

            loss_mean.backward()
            self.optim.step()
            self.optim.zero_grad()

            wandb.run.summary['step'] += 1

            losses.append(loss_mean.item())

            wandb.log({'loss_batch': loss_mean.item()}, step=wandb.run.summary['step'])

        return np.mean(losses)

    def get_policy(self):
        return policy.DeepPolicy(self.net)
