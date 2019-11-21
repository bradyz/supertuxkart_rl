from typing import Dict

import numpy as np
import torch
import pystk
import cv2


N_ACTIONS = int(2 ** 4)


class BasePolicy:
    def __call__(self, m: Dict):
        raise NotImplementedError("BasePolicy.__call__")


class DiscretePolicy(BasePolicy):
    def __init__(self, net, n_actions, eps):
        self.net = net
        self.net.eval()

        self.n_actions = n_actions
        self.eps = eps

    def __call__(self, s, v):
        with torch.no_grad():
            s = s.transpose(2, 0, 1)
            s = torch.FloatTensor(s).unsqueeze(0).cuda()
            m = torch.distributions.Categorical(logits=self.net(s))

        if np.random.rand() < self.eps:
            action_index = np.random.choice(list(range(self.n_actions)))
        else:
            action_index = m.sample().item()

        p = m.probs.squeeze()[action_index]
        p_action = (1 - self.eps) * p + self.eps / N_ACTIONS

        binary = bin(action_index).lstrip('0b').rjust(4, '0')

        action = pystk.Action()
        action.steer = int(binary[0] == '1') * -1.0 + int(binary[1] == '1') * 1.0
        action.acceleration = np.clip(5 + int(binary[2] == '1') * 20.0 - v, 0, 0.5)
        action.drift = binary[3] == '1'

        return action, action_index, p_action


class HumanPolicy(BasePolicy):
    def __call__(self, s, v):
        cv2.imshow('s', cv2.cvtColor(s, cv2.COLOR_BGR2RGB))

        key = cv2.waitKey(1)

        import time; time.sleep(1.0 / 10.0)

        action = pystk.Action()
        action.steer = int(key == 97) * -1.0 + int(key == 100) * 1.0
        action.acceleration = np.clip(5 + int(action.steer == 0) * 90.0 - v, 0, 0.5)

        return action, 0, 1.0
