import wandb
import numpy as np
import torch
import torch.nn.functional as F

from . import policy


class REINFORCE(object):
    def __init__(self, batch_size, lr, iterations, eps, importance_sampling, device, **kwargs):
        self.batch_size = batch_size
        self.lr = lr
        self.iterations = iterations
        self.importance_sampling = importance_sampling
        self.eps = eps

        self.device = device
        self.n_actions = 16

        self.net = Network(self.n_actions)
        self.net.to(device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def get_policy(self, epoch):
        return policy.DiscretePolicy(
                self.net,
                n_actions=self.n_actions,
                eps=np.clip((1 - epoch / 100) * self.eps, 0.0, 1.0))

    def train(self, replay):
        self.net.to(self.device)
        self.net.train()

        losses = list()

        for i in range(self.iterations):
            indices = np.random.choice(len(replay), self.batch_size, replace=False)
            s, _, a_i, p_a, _, _, R = replay[indices]

            s = torch.FloatTensor(s.transpose(0, 3, 1, 2)).to(self.device)
            a_i = torch.LongTensor(a_i).squeeze().to(self.device)
            p_a = torch.FloatTensor(p_a).to(self.device)
            R = torch.FloatTensor(R).squeeze().to(self.device)
            R = (R - R.mean()) / (R.std() + 1e-5)

            m = torch.distributions.Categorical(logits=self.net(s))
            log_p = m.log_prob(a_i)
            rho = 1.0

            if self.importance_sampling:
                rho = torch.exp(log_p) / p_a

            loss = (rho * -log_p * R).mean()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1e-2)
            self.optim.step()
            self.optim.zero_grad()

            losses.append(loss.item())

            wandb.run.summary['step'] += 1
            wandb.log({'batch/loss': loss.item()}, step=wandb.run.summary['step'])

        print(log_p[:8])
        print(R[:8])

        return {
                'epoch/loss': np.mean(losses),
                }


class Network(torch.nn.Module):
    def __init__(self, n_actions):
        super().__init__()

        # self.norm = torch.nn.BatchNorm2d(1)
        self.norm = lambda x: x / 255.0 - 0.5
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, 4, 2)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1)

        self.fc4 = torch.nn.Linear(4 * 4 * 64, 512)
        self.fc5 = torch.nn.Linear(512, n_actions)
        # self.bn = torch.nn.BatchNorm1d(n_actions)

    def forward(self, x):
        x = self.norm(x.mean(1, True))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        x = self.fc5(x)
        # x = self.bn(x)

        return x
