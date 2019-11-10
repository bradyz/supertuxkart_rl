from collections import deque
from typing import Dict

import pystk
import ray
import numpy as np

from . import controller, policy
from .replay_buffer import Data


def point_from_line(p, a, b):
    u = p - a
    u = np.float32([u[0], u[2]])

    v = b - a
    v = np.float32([v[0], v[2]])
    v_norm = v / np.linalg.norm(v)

    closest = u.dot(v_norm) * v_norm

    # import matplotlib.pyplot as plt; plt.ion()
    # plt.clf()
    # plt.axis('equal')
    # plt.plot([0, v[0]], [0, v[1]], 'r-')
    # plt.plot(u[0], u[1], 'bo')
    # plt.plot(closest[0], closest[1], 'ro')
    # plt.pause(0.01)

    return np.linalg.norm(u - closest)


class Rollout(object):
    def __init__(self, track):
        config = pystk.GraphicsConfig.ld()
        config.screen_width = 64
        config.screen_height = 64
        config.render_window = False

        pystk.init(config)

        race_config = pystk.RaceConfig()
        race_config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
        race_config.track = track
        race_config.step_size = 0.1
        race_config.render = True

        self.race = pystk.Race(race_config)
        self.race.start()
        self.race.step()

        self.track = pystk.Track()
        self.track.update()

    def rollout(
            self,
            policy: policy.BasePolicy,
            max_step: float = 100,
            frame_skip: int = 0,
            gamma: float = 1.0):
        self.race.restart()
        self.race.step(pystk.Action())
        self.track.update()

        result = list()

        state = pystk.WorldState()
        state.update()

        r_total = 0
        farthest = 0

        d = state.karts[0].distance_down_track
        s = np.array(self.race.render_data[0].image)

        off_track = deque(maxlen=20)
        no_progress = deque(maxlen=20)

        for it in range(max_step):
            # Early termination.
            if it > len(off_track) and ((all(no_progress) or all(off_track))):
                break

            v = np.linalg.norm(state.karts[0].velocity)
            action, action_i, p_action = policy(s, v)

            for _ in range(1 + frame_skip):
                self.race.step(action)
                self.track.update()

                state = pystk.WorldState()
                state.update()

            s_p = np.array(self.race.render_data[0].image)

            d_new = state.karts[0].distance_down_track
            node_idx = np.searchsorted(
                    self.track.path_distance[:, 1],
                    d_new % self.track.path_distance[-1, 1]) % len(self.track.path_nodes)
            a_b = self.track.path_nodes[node_idx]

            distance = point_from_line(state.karts[0].location, a_b[0], a_b[1])
            mult = int(distance < 8.0) * 2.0 - 1.0

            # Weight this?
            farthest = max(farthest, d_new)
            r_total = max(r_total, d_new * mult)
            r = np.clip(max(r_total - d, 0) + 0.5 * mult, -1.0, 1.0)
            no_progress.append(d_new < farthest)
            off_track.append(distance > 8.0)

            result.append(
                    Data(
                        s.copy(),
                        np.float32([action.steer, action.acceleration, action.drift]),
                        np.uint8([action_i]), np.float32([p_action]),
                        np.float32([r]), s_p.copy(),
                        np.float32([np.nan])))

            d = d_new
            s = s_p

        G = 0

        # Ugly.
        for i, data in enumerate(reversed(result)):
            G = data.r + gamma * G
            result[-(i + 1)] = Data(
                    data.s,
                    data.a, data.a_i, data.p_a,
                    data.r, data.sp,
                    np.float32([G]))

        return result, r_total

    def __del__(self):
        self.race.stop()
        self.race = None
        self.track = None

        pystk.clean()


@ray.remote(num_cpus=1, num_gpus=0.15)
class RayRollout(Rollout):
    pass


if __name__ == "__main__":
    rollout = Rollout()
    rollout.start()

    episode = rollout.rollout(
            # policy.RewardPolicy(Reward),
            policy.HumanPolicy(),
            controller.TuxController(),
            max_step=1000)
