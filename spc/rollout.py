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

        for it in range(max_step):
            # Autopilot.
            # birdview = self.map.draw_track(state.karts[0])['track']

            # ty, tx = policy(birdview)
            # world_target = self.map.to_world(tx, ty)
            # action = controller(state.karts[0], world_target)

            # import cv2
            # image = birdview[:,:,2]/50 + birdview[:,:,1]
            # cv2.imshow('image', image)
            # cv2.waitKey(1)

            s = np.uint8(self.race.render_data[0].image)

            # Network.
            action, action_i, p_action = policy(s)
            self.race.step(action)

            state = pystk.WorldState()
            state.update()

            # HACK: fix...
            d_new = state.karts[0].distance_down_track
            r = d_new - d
            r_list.append(r)
            r_total = max(r_total, d)
            d_new = d

            result.append(
                    Data(
                        s.copy(), np.uint8([action_i]), np.float32([p_action]),
                        np.array([r]), np.array([False])))

        G_list = list()
        G = 0

        # Hack.
        for r in r_list[::-1]:
            G = r + gamma * G
            G_list.insert(0, G)

        for i, data in enumerate(result):
            result[i] = Data(
                    data.s, data.a, np.float32([p_action]),
                    np.float32([G_list[i]]), data.done)

        return result, r_total

    def __del__(self):
        self.stop()

        pystk.clean()


@ray.remote(num_cpus=1, num_gpus=1.0 / 8.0)
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
