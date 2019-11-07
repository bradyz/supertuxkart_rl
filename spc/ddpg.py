import wandb
import numpy as np
import torch
import torch.nn.functional as F

from . import policy


N_ACTIONS = 2


class DDPG(object):
    def __init__(self, batch_size, lr, gamma, eps, device, **kwargs):
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.eps = eps

        self.device = device

        self.actor = Actor(N_ACTIONS)
        self.actor.to(device)

        self.actor_target = Actor(N_ACTIONS)
        self.actor_target.to(device)

        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        self.critic = Critic(N_ACTIONS)
        self.critic.to(device)

        self.critic_target = Critic(N_ACTIONS)
        self.critic_target.to(device)

        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        for u, v in zip(self.actor.parameters(), self.actor_target.parameters()):
            u.data.copy_(v.data)

        for u, v in zip(self.critic.parameters(), self.critic_target.parameters()):
            u.data.copy_(v.data)

    def train(self, replay, n_iterations=1000):
        self.actor.to(self.device)
        self.actor.train()

        self.critic.to(self.device)
        self.critic.train()

        losses_actor = list()
        losses_critic = list()

        import tqdm

        for i in tqdm.tqdm(range(n_iterations)):
            indices = np.random.choice(len(replay), self.batch_size)
            s, a, p_a, r, sp, R = replay[indices]

            s = torch.FloatTensor(s.transpose(0, 3, 1, 2))
            s = s.to(self.device)

            sp = torch.FloatTensor(sp.transpose(0, 3, 1, 2))
            sp = sp.to(self.device)

            a = torch.FloatTensor(a).squeeze()
            a = a.to(self.device)

            # PLEASE
            a = torch.stack(2 * [a], 1)

            r = torch.FloatTensor(r).squeeze()
            r = r.to(self.device)

            p_a = torch.FloatTensor(p_a).squeeze()
            p_a = p_a.to(self.device)

            R = torch.FloatTensor(R).squeeze()
            R = R.to(self.device)

            a_hat_target = self.actor_target(s)
            q_hat_target = self.critic_target(sp, a_hat_target)

            y = r + self.gamma * q_hat_target
            y_hat = self.critic(s, a)

            critic_loss = ((y_hat - y) ** 2).mean()

            self.optim_actor.zero_grad()
            self.optim_critic.zero_grad()
            critic_loss.backward()
            self.optim_critic.step()

            a_hat = self.actor(s)
            q_hat = self.critic(s, a_hat)

            actor_loss = -q_hat.mean()

            self.optim_actor.zero_grad()
            self.optim_critic.zero_grad()
            actor_loss.backward()
            self.optim_actor.step()

        tau = 0.001

        for u, v in zip(self.actor.parameters(), self.actor_target.parameters()):
            v.data.copy_(tau * v.data + (1 - tau) * u.data)

        for u, v in zip(self.critic.parameters(), self.critic_target.parameters()):
            v.data.copy_(tau * v.data + (1 - tau) * u.data)

        return metrics

    def get_policy(self):
        return policy.DeepPolicy(self.actor)


class Critic(torch.nn.Module):
    def __init__(self, n_actions):
        super().__init__()

        self.norm = torch.nn.BatchNorm2d(3)
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, 4, 2)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1)

        self.fc4 = torch.nn.Linear(4 * 4 * 64, 512)
        self.fc5 = torch.nn.Linear(512+n_actions, 1)

    def forward(self, x, a):
        x = self.norm(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        x = torch.cat([x, a], 1)

        x = self.fc5(x)

        return x


class Actor(torch.nn.Module):
    def __init__(self, n_actions):
        super().__init__()

        self.norm = torch.nn.BatchNorm2d(3)
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, 4, 2)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1)

        self.fc4 = torch.nn.Linear(4 * 4 * 64, 512)
        self.fc5 = torch.nn.Linear(512, n_actions)

    def forward(self, x):
        x = self.norm(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        x = self.fc5(x)
        x = torch.tanh(x)
        x[:,1] = (x[:,1] / 2.0) + 0.5

        return x
