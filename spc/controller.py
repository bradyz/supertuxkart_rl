from pystk import Kart, Action
import numpy as np


class BaseController:
    def act(self, state: Kart, target):
        raise NotImplemented

    def __call__(self, state: Kart, target):
        return self.act(state, target)


class TuxController(BaseController):
    def __init__(self, drift_threshold=0.1, min_drift_speed=5):
        self.drift_threshold = drift_threshold
        self.min_drift_speed = min_drift_speed
        self.current_drift = 0
        self.last_drift = 0

    def act(self, state: Kart, target):
        a = Action()
        t = np.asarray(target)
        x = np.asarray(state.location[::2])
        dx = np.asarray(state.front[::2])-x
        v = np.asarray(state.velocity[::2])

        t_dx = t-x
        target_angle = np.arctan2(*dx[::-1]) - np.arctan2(*t_dx[::-1])
        if target_angle > np.pi: target_angle -= 2*np.pi
        if target_angle <-np.pi: target_angle += 2*np.pi

        steer = target_angle / state.max_steer_angle
        a.steer = np.clip(steer, -1, 1)
        a.acceleration = 1

        drift = int(steer < -self.drift_threshold) - int(steer > self.drift_threshold)
        a.drift = self.current_drift == drift and drift != 0 # and kart_state.speed > min_drift_speed
        # If we need to turn sharply release drift for one timestep, and drift again (will turn the kart)
        if abs(steer) > 1.5:
            # Release the drift if we are at a too sharp of an angle
            a.drift = not self.last_drift # and abs(steer) < 2.5
        self.last_drift = a.drift
        self.current_drift = drift
        return a

