from typing import Dict

import numpy as np
import torch


class BasePolicy:
    def __call__(self, m: Dict):
        raise NotImplementedError("BasePolicy.__call__")


class RewardPolicy(BasePolicy):
    def __init__(self, reward_fun):
        self.reward_fun = reward_fun

    def __call__(self, m: Dict):
        r = self.reward_fun(m)

        return np.unravel_index(np.argmax(r), r.shape)


class DeepNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, 4, 2)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1)

        self.fc4 = torch.nn.Linear(6 * 6 * 64, 512)
        self.fc5 = torch.nn.Linear(512, 64)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        x = self.fc5(x)

        return x


class DeepPolicy(BasePolicy):
    def __init__(self, net):
        self.net = net

    def __call__(self, s):
        s = torch.FloatTensor(s.transpose(0, 3, 1, 2))
        s = s.to(self.net.device)

        prob = net(s)
        import pdb; pdb.set_trace()
