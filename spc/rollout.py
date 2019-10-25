import time
import pystk
from typing import Tuple, Dict, Set
from . import controller, policy
import ray
from .track_map import Map, colored_map
import numpy as np


class Reward:
    def __call__(self, track_map):
        return track_map[:, :, 1] + track_map[:, :, 2] / 50.


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

    def update(self, kart_info, target, action, I=None):
        self.silent_update(kart_info, target, action)
        self.plot(kart_info, I)


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
            # config.track = "zengarden"
            config.step_size = 0.1

        if self.race is not None:
            self.race.stop()
            del self.race

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

    def rollout(self, p: policy.BasePolicy, c: controller.BaseController, max_step: float = 100, restart: bool = True,
                return_data: Set[str] = {}):
        """
        :param return_data: what data should we return? 'action', 'image', 'map', 'state', 'track'
        :return:
        """
        import collections
        Data = collections.namedtuple('Data', 'action image map state track')
        assert self.race is not None, "You need to start the case before the rollout"

        if restart:
            self.race.restart()

        next_action = pystk.Action()
        result = []
        for it in range(max_step):
            self.race.step(next_action)

            state = pystk.WorldState()
            state.update()

            drawn_map = self.map.draw_track(state.karts[0])
            ty, tx = p(drawn_map['track'])
            world_target = self.map.to_world(tx, ty)
            next_action = c(state.karts[0], world_target)

            # Handle the return data
            t, i, m, s, a = None, None, None, None, None
            if 'track' in return_data and it == 0:
                t = self.track
            if 'action' in return_data:
                a = next_action
            if 'image' in return_data:
                i = 1*np.asarray(self.race.render_data[0].image)
            if 'map' in return_data:
                m = drawn_map['track']
            if 'state' in return_data:
                s = state

            result.append(Data(action=a, image=i, map=m, state=s, track=t))
        return result

    def __del__(self):
        if self.race is not None:
            self.race.stop()
            del self.race

        pystk.clean()


@ray.remote
class RayRollout(Rollout):
    pass


if __name__ == "__main__":
    ray.init(logging_level=40)  # logging.ERROR

    rollouts = [RayRollout.remote() for _ in range(8)]
    # rollout.start()
    for rollout in rollouts:
        ray.get(rollout.start.remote())

    tick = time.time()

    ros = [rollout.rollout.remote(policy.RewardPolicy(Reward()), controller.TuxController(), return_data={'image', 'map', 'state', 'track'}, max_step=100) for rollout in rollouts]

    ros = [ray.get(ro) for ro in ros]

    clock = time.time() - tick

    print(clock)
    print(sum(map(len, ros)) / clock)

    for j in range(len(ros)):

        viz = TrackViz(ros[j][0].track)
        for a, i, m, s, t in ros[j]:
            viz.plot(s.karts[0], i)
