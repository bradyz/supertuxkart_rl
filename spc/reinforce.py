import wandb
import numpy as np
import torch
import torch.nn.functional as F

from . import policy


class REINFORCE(object):
    def __init__(self, batch_size, lr, gamma, importance_sampling, device, **kwargs):
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.importance_sampling = importance_sampling

        self.device = device

        self.net = Network()
        self.net.to(device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def train(self, replay, n_iterations=1000):
        self.net.to(self.device)
        self.net.train()

        losses = list()

        for i in range(n_iterations):
            indices = np.random.choice(len(replay), self.batch_size)
            s, a, a_i, p_a, r, sp, R = replay[indices]

            s = torch.FloatTensor(s.transpose(0, 3, 1, 2))
            s = s.to(self.device)

            a_i = torch.FloatTensor(a_i).squeeze()
            a_i = a_i.to(self.device)

            R = torch.FloatTensor(R)
            R = R.to(self.device)
            R = (R - R.mean()) / (R.std() + 1e-7)

            p_a = torch.FloatTensor(p_a)
            p_a = p_a.to(self.device)

            m = torch.distributions.Categorical(logits=self.net(s))
            log_p = m.log_prob(a_i)
            rho = 1.0

            if self.importance_sampling:
                rho = torch.exp(log_p) / p_a

            loss = -(rho * R * log_p).sum(1)
            loss_mean = loss.mean()

            loss_mean.backward()
            self.optim.step()
            self.optim.zero_grad()

            wandb.run.summary['step'] += 1

            losses.append(loss_mean.item())

            wandb.log({'loss_batch': loss_mean.item()}, step=wandb.run.summary['step'])

        metrics = {
                'epoch/loss': np.mean(losses),
                }

        return metrics

    def get_policy(self):
        return policy.DeepPolicy(self.net)


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.norm = torch.nn.BatchNorm2d(1)
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, 4, 2)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1)

        self.fc4 = torch.nn.Linear(4 * 4 * 64, 512)
        self.fc5 = torch.nn.Linear(512, 8)

    def forward(self, x):
        x = self.norm(x.mean(1, True))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        x = self.fc5(x)

        return x
