import pystk
import wandb
import numpy as np
import torch
import torch.nn.functional as F

from .policy import BasePolicy


N_ACTIONS = 3


class DDPG(object):
    def __init__(self, batch_size, iterations, lr, lr_1, gamma, eps, tau, device, **kwargs):
        self.batch_size = batch_size
        self.lr = lr
        self.lr_1 = lr_1
        self.gamma = gamma
        self.eps = eps
        self.tau = tau
        self.iterations = iterations

        self.device = device

        self.actor = Actor(N_ACTIONS)
        self.actor.to(self.device)
        self.actor_target = Actor(N_ACTIONS)
        self.actor_target.to(self.device)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        self.critic = Critic(N_ACTIONS)
        self.critic.to(self.device)
        self.critic_target = Critic(N_ACTIONS)
        self.critic_target.to(self.device)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_1)

        for u, v in zip(self.actor.parameters(), self.actor_target.parameters()):
            u.data.copy_(v.data)

        for u, v in zip(self.critic.parameters(), self.critic_target.parameters()):
            u.data.copy_(v.data)

    def train(self, replay):
        self.actor.to(self.device)
        self.actor.train()

        self.critic.to(self.device)
        self.critic.train()

        losses_actor = list()
        losses_critic = list()

        for i in range(self.iterations):
            indices = np.random.choice(len(replay), self.batch_size)
            s, a, _, _, r, sp, R, done = replay[indices]

            s = torch.FloatTensor(s.transpose(0, 3, 1, 2))
            s = s.to(self.device)

            sp = torch.FloatTensor(sp.transpose(0, 3, 1, 2))
            sp = sp.to(self.device)

            a = torch.FloatTensor(a).squeeze()
            a = a.to(self.device)

            r = torch.FloatTensor(r).squeeze()
            r = r.to(self.device)

            R = torch.FloatTensor(R).squeeze()
            R = R.to(self.device)

            done = torch.FloatTensor(done).squeeze()
            done = done.to(self.device)

            a_hat_target = self.actor_target(s)
            q_hat_target = self.critic_target(sp, a_hat_target).squeeze()

            y = r + (1.0 - done) * self.gamma * q_hat_target
            y_hat = self.critic(s, a).squeeze()

            # loss_critic = ((y_hat - y) ** 2).mean() + ((y_hat - R) ** 2).mean()
            loss_critic = ((y_hat - y) ** 2).mean()

            self.optim_actor.zero_grad()
            self.optim_critic.zero_grad()
            loss_critic.backward()
            self.optim_critic.step()

            a_hat = self.actor(s)
            q_hat = self.critic(s, a_hat).squeeze()

            loss_actor = -q_hat.mean()

            self.optim_actor.zero_grad()
            self.optim_critic.zero_grad()
            loss_actor.backward()
            self.optim_actor.step()

            for u, v in zip(self.actor_target.parameters(), self.actor.parameters()):
                u.data.copy_((1 - self.tau) * u.data + self.tau * v.data)

            for u, v in zip(self.critic_target.parameters(), self.critic.parameters()):
                u.data.copy_((1 - self.tau) * u.data + self.tau * v.data)

            losses_critic.append(loss_critic.item())
            losses_actor.append(loss_actor.item())

            wandb.run.summary['step'] += 1
            wandb.log({
                'batch/actor': loss_actor.item(),
                'batch/critic': loss_critic.item(),
                },
                step=wandb.run.summary['step'])

        return {
                'epoch/actor': np.mean(losses_actor),
                'epoch/critic': np.mean(losses_critic),
                }

    def get_policy(self, epoch):
        noise = np.clip((1 - epoch / 5000) * self.eps, 0.0, 1.0)
        wandb.log({'epoch/eps': noise}, step=wandb.run.summary['step'])

        return ContinuousPolicy(self.actor, noise)


class Critic(torch.nn.Module):
    def __init__(self, n_actions):
        super().__init__()

        self.norm = torch.nn.BatchNorm2d(3)
        self.norm_a = torch.nn.BatchNorm1d(n_actions)
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

        x = torch.cat([x, self.norm_a(a)], 1)
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

        x[:, 0] = torch.tanh(x[:, 0])
        x[:, 1] = torch.sigmoid(x[:, 1]) * 5.0
        x[:, 2] = torch.sigmoid(x[:, 2])

        return x


def ou(x, mu, theta, sigma):
    return theta * (mu - x) + sigma * np.random.randn()


class ContinuousPolicy(BasePolicy):
    def __init__(self, net, noise):
        self.net = net
        self.net.eval()
        self.noise = noise

    def __call__(self, s, v):
        with torch.no_grad():
            s = s.transpose(2, 0, 1)
            s = torch.FloatTensor(s).unsqueeze(0).cuda()
            a = self.net(s).squeeze()

        steer = a[0].item()
        velocity = a[1].item()
        drift = a[2].item()

        action = [
                steer + self.noise * ou(steer, 0.0, 0.2, 0.2),
                velocity + self.noise * ou(velocity, 2.5, 0.5, 2.5),
                drift + self.noise * np.random.randn() * 0.50]

        return action, -1, 1.0
