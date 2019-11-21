import numpy as np
import torch
import torch.nn.functional as F

from . import policy


N_ACTIONS = 1


class SAC(object):
    def __init__(self, batch_size, lr, gamma, eps, alpha, tau, device, **kwargs):
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.alpha = alpha
        self.tau = tau                          # 1e-2

        self.device = device

        # Q-networks.
        self.q_1 = Critic(N_ACTIONS)
        self.q_2 = Critic(N_ACTIONS)
        self.optim_q = torch.optim.Adam(
                list(self.q_1.parameters()) + list(self.q_2.parameters()),
                lr=self.lr)

        # Value function.
        self.v = Value()
        self.v_target = Value()
        self.optim_v = torch.optim.Adam(self.v.parameters(), lr=self.lr)

        # Actor.
        self.pi = Actor(N_ACTIONS)
        self.optim_pi = torch.optim.Adam(self.pi.parameters(), lr=self.lr)

        for u, v in zip(self.v_target.parameters(), self.v.parameters()):
            u.data.copy_(v.data)

    def train(self, replay, n_iterations=1000):
        self.pi.to(self.device)
        self.pi.train()

        self.q_1.to(self.device)
        self.q_1.train()
        self.q_2.to(self.device)
        self.q_2.train()

        self.v.to(self.device)
        self.v.train()

        self.v_target.to(self.device)
        self.v_target.train()

        import tqdm

        for i in tqdm.tqdm(range(n_iterations)):
            indices = np.random.choice(len(replay), self.batch_size)
            s, a, a_i, p_a, r, sp, R = replay[indices]

            s = torch.FloatTensor(s.transpose(0, 3, 1, 2))
            s = s.to(self.device)

            sp = torch.FloatTensor(sp.transpose(0, 3, 1, 2))
            sp = sp.to(self.device)

            a = torch.FloatTensor(a).squeeze()
            a = a.to(self.device)[:,:2]

            r = torch.FloatTensor(r).squeeze()
            r = r.to(self.device)

            p_a = torch.FloatTensor(p_a).squeeze()
            p_a = p_a.to(self.device)

            R = torch.FloatTensor(R).squeeze()
            R = R.to(self.device)

            R_est = r + self.gamma * self.v_target(sp)
            a_tilde, log_p_a_tilde = self.pi(s)

            V_est = min(self.q_1(s, a_tilde), self.q_2(s, a_tilde)) - self.alpha * log_p_a_tilde

            q_loss = ((self.q_1(s, a) - R_est) ** 2).sum(-1).mean()
            q_loss += ((self.q_2(s, a) - R_est) ** 2).sum(-1).mean()
            q_loss.backward()
            self.optim_q.step()
            self.optim_q.zero_grad()

            v_loss = ((self.v(s) - V_est) ** 2).sum(-1).mean()
            v_loss.backward()
            self.optim_v.step()
            self.optim_v.zero_grad()

            pi_loss = -V_est
            self.optim_pi.zero_grad()
            pi_loss.backward()
            self.optim_pi.step()
            self.optim_pi.zero_grad()

        for u, v in zip(self.v_pi.parameters(), self.v.parameters()):
            v.data.copy_(self.tau * v.data + (1 - self.tau) * u.data)

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


class Value(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.norm = torch.nn.BatchNorm2d(3)
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, 4, 2)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1)

        self.fc4 = torch.nn.Linear(4 * 4 * 64, 512)
        self.fc5 = torch.nn.Linear(512, 1)

    def forward(self, x):
        x = self.norm(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
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
        self.fc5 = torch.nn.Linear(512, 2 * n_actions)

    def forward(self, x, log_probs=False):
        x = self.norm(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        x = self.fc5(x)

        mus = x[:, :x.shape[-1] // 2].squeeze()
        sigmas = x[:, x.shape[-1] // 2:].squeeze()
        a = torch.tanh(mus + sigmas * torch.randn(sigmas.shape).to(x.device))
        log_p_a = torch.distributions.Normal(
                mus, sigmas).log_probs(a) - torch.log(1 - a ** 2)

        return a, log_p_a
