from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import pystk


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

        self.fc4 = torch.nn.Linear(4 * 4 * 64, 512)
        self.fc5 = torch.nn.Linear(512, int(2 ** 3))

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
        self.net.eval()

    def __call__(self, s):
        if np.random.rand() < 0.25:
            action_index = np.random.choice(list(range(8)))
        else:
            # HACK: deterministic
            with torch.no_grad():
                s = s.transpose(2, 0, 1)
                s = torch.FloatTensor(s).unsqueeze(0).cuda()

                m = torch.distributions.Categorical(logits=self.net(s))
                action_index = m.sample().item()

        binary = bin(action_index).lstrip('0b').rjust(3, '0')

        action = pystk.Action()
        action.steer = int(binary[0] == '1') * -1.0 + int(binary[1] == '1') * 1.0
        action.acceleration = int(binary[2] == '1') * 0.25

        return action, action_index


class HumanPolicy(BasePolicy):
    def __call__(self, s):
        import cv2

        cv2.imshow('s', cv2.cvtColor(s, cv2.COLOR_BGR2RGB))
        key = cv2.waitKey(10)

        action = pystk.Action()
        action.steer = int(key == 97) * -1.0 + int(key == 100) * 1.0
        action.acceleration = 0.1

        return action, 0
