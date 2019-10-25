import time

import numpy as np
import ray

from .rollout import RayRollout, Reward
from .replay_buffer import ReplayBuffer
from . import controller, policy


def get_samples(agent, max_step=1000, n_workers=8):
    tick = time.time()

    rollouts = [RayRollout.remote() for _ in range(n_workers)]

    [ray.get(rollout.start.remote()) for rollout in rollouts]

    ros = list()

    for rollout in rollouts:
        ros.append(rollout.rollout.remote(
                agent, controller.TuxController(),
                max_step=max_step))

    ros = [ray.get(ro) for ro in ros]

    clock = time.time() - tick

    print('FPS: %.2f' % (sum(map(len, ros)) / clock))
    print('AFPS: %.2f' % (np.mean(list(map(len, ros))) / clock))

    return ros


def main():
    replay = ReplayBuffer()
    agent = policy.RewardPolicy(Reward())

    for _ in range(5):
        for rollout in get_samples(agent):
            for data in rollout:
                replay.add(data)

            print(len(replay))

    s, a, sp, r, done = replay[[1, 0]]


if __name__ == '__main__':
    ray.init(logging_level=40)

    main()
