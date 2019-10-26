from typing import Dict

import pystk
import ray
import numpy as np

from . import controller, policy
from .track_map import Map
from .replay_buffer import Data


class Reward:
    def __call__(self, track_map):
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

    def start(self, config: pystk.RaceConfig = None, map_config: Dict = dict(world_size=(50, 50), max_offset=50)):
        if config is None:
            config = pystk.RaceConfig()
            config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
            config.track = np.random.choice(["lighthouse", "zengarden"])
            config.track = "zengarden"
            config.step_size = 0.05

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
            restart: bool = True):

        assert self.race is not None, "You need to start the case before the rollout"

        if restart:
            self.race.restart()
            self.race.step(pystk.Action())

        result = list()

        state = pystk.WorldState()
        state.update()

        s = np.uint8(self.race.render_data[0].image)

        r_list = list()

        for it in range(max_step):
            # Autopilot.
            # ty, tx = policy(self.map.draw_track(state.karts[0])['track'])
            # world_target = self.map.to_world(tx, ty)
            # action = controller(state.karts[0], world_target)

            # Network.
            action, action_i = policy(s)

            # HACK: fix...
            r = np.linalg.norm(state.karts[0].velocity)
            r_list.append(r if r <= 1 else 0)

            self.race.step(action)

            state = pystk.WorldState()
            state.update()

            sp = np.uint8(self.race.render_data[0].image)

            result.append(
                    Data(
                        s.copy(), np.uint8([action_i]), sp.copy(),
                        np.array([r]), np.array([False])))

            s = sp

        G_list = list()
        G = 0
        gamma = 0.9

        for r in r_list[::-1]:
            G = r + gamma * G
            G_list.insert(0, G)

        for i, data in enumerate(result):
            result[i] = Data(
                    data.s, data.a,
                    np.float32([G_list[i]]), data.sp, data.done)

        return result

    def __del__(self):
        self.stop()

        pystk.clean()


@ray.remote(num_gpus=0.25)
class RayRollout(Rollout):
    pass


if __name__ == "__main__":
    rollout = Rollout()
    rollout.start()

    episode = rollout.rollout(
            policy.RewardPolicy(Reward()),
            controller.TuxController(),
            max_step=100)

    for s, a, g, sp, done in episode:
        print(g)
