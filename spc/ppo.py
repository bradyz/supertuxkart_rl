import wandb
import numpy as np
import torch
import torch.nn.functional as F

from . import policy


class PPO(object):
    def __init__(self, batch_size, lr, iterations, eps, gamma, clip, device, **kwargs):
        self.batch_size = batch_size
        self.lr = lr
        self.iterations = iterations
        self.gamma = gamma
        self.clip = clip
        self.eps = eps

        self.device = device
        self.n_actions = 16

        self.actor = Network(self.n_actions)
        self.actor.to(device)

        self.critic = Network(1)
        self.critic.to(device)

        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

    def get_policy(self, epoch):
        return policy.DiscretePolicy(
                self.actor,
                n_actions=self.n_actions,
                eps=np.clip((1 - epoch / 100) * self.eps, 0.0, 1.0))

    def train(self, replay):
        self.actor.to(self.device)
        self.actor.train()

        self.critic.to(self.device)
        self.critic.train()

        losses_actor = list()
        losses_critic = list()

        for i in range(self.iterations):
            indices = np.random.choice(len(replay), self.batch_size)
            s, a, a_i, p_a, r, sp, R = replay[indices]

            s = torch.FloatTensor(s.transpose(0, 3, 1, 2)).to(self.device)
            sp = torch.FloatTensor(sp.transpose(0, 3, 1, 2)).to(self.device)
            a_i = torch.LongTensor(a_i).squeeze().to(self.device)
            r = torch.FloatTensor(r).squeeze().to(self.device)
            p_a = torch.FloatTensor(p_a).squeeze().to(self.device)
            R = torch.FloatTensor(R).squeeze().to(self.device)
            # R = (R - R.mean()) / (R.std() + 1e-7)

            m = torch.distributions.Categorical(logits=self.actor(s))
            log_p = m.log_prob(a_i)

            V_s = self.critic(s).squeeze()

            # V_sp = self.critic(sp).squeeze()
            # A = (r + self.gamma * V_sp) - V_s

            A = R - V_s
            A = (A - A.mean()) / (A.std() + 1e-7)
            A = A.detach()

            rho = torch.exp(log_p) / p_a
            objective = torch.min(
                    rho * A,
                    torch.clamp(rho, 1 - self.clip, 1 + self.clip) * A)

            loss_actor = -(objective + 0.01 * m.entropy()).mean()
            loss_critic = ((V_s - R) ** 2).mean()

            loss_critic.backward()
            self.optim_critic.step()
            self.optim_critic.zero_grad()

            loss_actor.backward()
            self.optim_actor.step()
            self.optim_actor.zero_grad()

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



class Network(torch.nn.Module):
    def __init__(self, n_outputs):
        super().__init__()

        self.norm = torch.nn.BatchNorm2d(1)
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, 4, 2)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1)

        self.fc4 = torch.nn.Linear(4 * 4 * 64, 512)
        self.fc5 = torch.nn.Linear(512, n_outputs)

    def forward(self, x):
        x = self.norm(x.mean(1, True))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        x = self.fc5(x)

        return x
