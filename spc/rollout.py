from collections import deque
from typing import Dict

import pystk
import ray
import numpy as np

from . import controller, policy
from .track_map import Map
from .replay_buffer import Data


class Reward:
    def __call__(self, track_map):
        import pdb; pdb.set_trace()

        return track_map[:, :, 1] + track_map[:, :, 2] / 50.


def action_to_numpy(action):
    return np.float32([
        action.steer, action.acceleration, action.brake,
        action.drift, action.nitro])


class_color = np.array([
    0xffffff,  # None
    0x4e9a06,  # Kart
    0x2e3436,  # Track
    0xeeeeec,  # Background
    0x204a87,  # Pickup
    0xa40000,  # Bomb
    0xce5c00,  # Object
    0x5c3566,  # Projectile
], dtype='>u4').view(np.uint8).reshape((-1, 4))[:, 1:]


def _c(i, m):
    return m[i % len(m)]


def semantic_seg(instance, colorize: bool = True):
    L = (np.array(instance) >> 24) & 0xff
    if colorize:
        return _c(L, class_color)
    return L


class TrackViz:
    def __init__(self, track, visible_range=50):
        from pylab import figure
        self.track = track
        self.fig = figure()
        self.ax = self.fig.add_subplot(121)
        self.I_ax = self.fig.add_subplot(122)
        self.kart_loc = []
        self.front_loc = []
        self.target_loc = []
        self.visible_range = visible_range

    def silent_update(self, kart_info, target, action):
        self.kart_loc.append(kart_info.location[::2])
        self.front_loc.append(kart_info.front[::2])
        self.target_loc.append(target)

    def plot(self, kart_info, I=None):
        from pylab import pause
        self.ax.clear()
        self.ax.plot(*self.track.path_nodes[:, 0, ::2].T, '-', 'r')
        self.ax.plot(*self.track.path_nodes[:, 1, ::2].T, '-', 'g')

        self.ax.plot(*np.array(self.kart_loc).T, '-*', 'b')
        self.ax.plot(*np.array(self.front_loc).T, '-*', 'g')
        self.ax.plot(*np.array(self.target_loc).T, '-*', 'r')

        self.ax.set_xlim(kart_info.location[0] - self.visible_range, kart_info.location[0] + self.visible_range)
        self.ax.set_ylim(kart_info.location[2] - self.visible_range, kart_info.location[2] + self.visible_range)

        if I is not None:
            self.I_ax.clear()
            self.I_ax.imshow(I)

        self.fig.show()
        self.fig.canvas.draw()
        pause(1e-3)

    def update(self, kart_info, target, action, image=None):
        self.silent_update(kart_info, target, action)
        self.plot(kart_info, image)


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
    def __init__(self, config: pystk.GraphicsConfig = None):
        if config is None:
            config = pystk.GraphicsConfig.ld()
            config.screen_width = 64
            config.screen_height = 64

        pystk.init(config)

        self.config = None
        self.race = None
        self.track = None
        self.map = None

    def start(
            self,
            config: pystk.RaceConfig = None,
            map_config: Dict = dict(world_size=(50, 50), max_offset=50),
            track: str = 'lighthouse'):
        if config is None:
            config = pystk.RaceConfig()
            config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
            config.track = track
            config.step_size = 0.1

        self.stop()

        self.config = config

        self.race = pystk.Race(config)
        self.race.start()

        self.track = pystk.Track()
        self.track.update()

        self.map = Map(self.track, **map_config)

    def stop(self):
        if self.race is not None:
            self.race.stop()

            del self.race

        self.config = None
        self.race = None
        self.track = None
        self.map = None

    def rollout(
            self,
            policy: policy.BasePolicy,
            controller: controller.BaseController,
            max_step: float = 100,
            frame_skip: int = 2,
            restart: bool = True,
            gamma: float = 1.0):

        assert self.race is not None, "You need to start the case before the rollout"

        if restart:
            self.race.restart()
            self.race.step(pystk.Action())

        result = list()

        state = pystk.WorldState()
        state.update()

        r_list = list()
        r_total = 0

        d = state.karts[0].distance_down_track
        s = np.uint8(self.race.render_data[0].image)

        off_track = deque(maxlen=50)
        velocity = deque(maxlen=50)

        for it in range(max_step * frame_skip):
            # Autopilot.
            # birdview = self.map.draw_track(state.karts[0])['track']

            # ty, tx = policy(birdview)
            # world_target = self.map.to_world(tx, ty)
            # action = controller(state.karts[0], world_target)

            # import cv2
            # image = birdview[:,:,2]/50 + birdview[:,:,1]
            # cv2.imshow('image', image)
            # cv2.waitKey(1)

            # Network.
            action, action_i, p_action = policy(s)
            # s = np.uint8(self.race.render_data[0].image)
            # lanes = (semantic_seg(self.race.render_data[0].instance, False) == 2).mean()

            # # Network.
            # action, action_i = policy(s)

            # # HACK: fix...
            # r = np.linalg.norm(state.karts[0].velocity) + 1.0 * lanes
            # r_list.append(r)
            # r_total += r

            self.race.step(action)

            state = pystk.WorldState()
            state.update()

            s_p = np.uint8(self.race.render_data[0].image)

            # HACK: fix...
            d_new = state.karts[0].distance_down_track

            node_idx = np.searchsorted(
                    self.track.path_distance[:, 1],
                    d_new % self.track.path_distance[-1, 1]) % len(self.track.path_nodes)
            a_b = self.track.path_nodes[node_idx]

            distance = point_from_line(state.karts[0].location, a_b[0], a_b[1])
            mult = int(distance < 5.0) * 2.0 - 1.0

            r = (d_new - d)  * mult
            r_list.append(r)
            r_total = max(r_total, d_new * mult)
            d = d_new

            velocity.append(np.linalg.norm(state.karts[0].velocity))
            off_track.append(distance > 5)

            if it > 100 and ((sum(velocity) / len(velocity) < 1.0 or all(off_track))):
                break

            if it % frame_skip == 0:
                result.append(
                        Data(
                            s.copy(),
                            np.float32([action.steer, action.acceleration, action.drift]),
                            np.uint8([action_i]), np.float32([p_action]),
                            np.float32([r]), s_p.copy(),
                            np.array([r])))

            s = s_p

        G_list = list()
        G = 0

        # Hack.
        for r in r_list[::-1]:
            G = r + gamma * G
            G_list.insert(0, G)

        for i, data in enumerate(result):
            result[i] = Data(
                    data.s,
                    data.a, data.a_i, data.p_a,
                    data.r, data.sp,
                    np.float32([G_list[i]]))

        return result, r_total

    def __del__(self):
        self.stop()

        pystk.clean()


@ray.remote(num_cpus=1, num_gpus=0.2)
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

    for s, a, g, done in episode:
        print(g)
