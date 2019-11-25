from collections import deque

import pystk
import ray
import numpy as np

from . import policy
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


def get_distance(d_new, d, track_length):
    if abs(d_new - d) > 100:
        sign = float(d_new < d) * 2 - 1
        d_new, d = min(d_new, d), max(d_new, d)

        return sign * ((d_new - d) % track_length)

    return d_new - d


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
        d = state.karts[0].distance_down_track
        s = np.array(self.race.render_data[0].image)

        off_track = deque(maxlen=20)
        traveled = deque(maxlen=50)

        for it in range(max_step):
            # Early termination.
            if it > 20 and (np.median(traveled) < 0.05 or all(off_track)):
                break

            v = np.linalg.norm(state.karts[0].velocity)
            action, action_i, p_action = policy(s, v)

            if isinstance(action, pystk.Action):
                action_raw = [action.steer, action.acceleration, action.drift]
            else:
                action_raw = action
                action = pystk.Action()
                action.steer = action_raw[0]
                action.acceleration = action_raw[1] - v
                action.drift = action_raw[2] > 0.5

            for _ in range(1 + frame_skip):
                self.race.step(action)
                self.track.update()

                state = pystk.WorldState()
                state.update()

            s_p = np.array(self.race.render_data[0].image)

            d_new = min(state.karts[0].distance_down_track, d + 5.0)
            node_idx = np.searchsorted(
                    self.track.path_distance[:, 1],
                    d_new % self.track.path_distance[-1, 1]) % len(self.track.path_nodes)
            a_b = self.track.path_nodes[node_idx]

            distance = point_from_line(state.karts[0].location, a_b[0], a_b[1])
            mult = int(distance < 6.0) * 2.0 - 1.0
            distance_traveled = get_distance(d_new, d, self.track.path_distance[-1, 1])
            gain = distance_traveled if distance_traveled > 0 else 0

            traveled.append(gain)
            off_track.append(distance > 6.0)

            r_total = max(r_total, d_new * mult)
            r = np.clip(0.5 * max(mult * gain, 0) + 0.5 * mult, -1.0, 1.0)

            result.append(
                    Data(
                        s.copy(),
                        np.float32(action_raw),
                        np.uint8([action_i]), np.float32([p_action]),
                        np.float32([r]), s_p.copy(),
                        np.float32([np.nan]),
                        np.float32([0])))

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
                    np.float32([G]),
                    np.float32([i == 0]))

        # HACK PLEASE REMEMBER THIS
        return result[4:], r_total / self.track.path_distance[-1, 1]

    def __del__(self):
        self.race.stop()
        self.race = None
        self.track = None

        pystk.clean()


@ray.remote(num_cpus=1, num_gpus=0.15)
class RayRollout(Rollout):
    pass


if __name__ == "__main__":
    rollout = Rollout('lighthouse')

    episode = rollout.rollout(
            policy.HumanPolicy(),
            max_step=1000000)
