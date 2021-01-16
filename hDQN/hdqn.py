from .hdqn_models import MetaControllerModel, ControllerModel
import gym
import numpy as np
import torch
import cv2


class MetaController:
    def __init__(self, n_options):
        self.model = MetaControllerModel(n_options=n_options, device='cuda')


class Controller:
    def __init__(self, action_space):
        self.action_space = action_space
        self.model = ControllerModel(n_actions=action_space.n, device='cuda')

    def act(self, state):
        return self.action_space.sample()


class hDQN:

    def __init__(self):
        self.env = gym.make("MontezumaRevengeNoFrameskip-v4")
        self.meta_controller = MetaController(n_options=3)
        self.controller = Controller(action_space=self.env.action_space)

    def _process_state(self, state):
        state = np.transpose(state, (2, 0, 1))
        return state

    def run(self, n_steps):
        self.env.reset()
        # get the observation from the ale buffer (this is faster than converting the color image to grayscale)
        prev_env_state = self.env.ale.getScreenGrayscale()
        prev_state = self._process_state(prev_env_state)

        for i in range(n_steps):

            cv2.imshow("render", prev_env_state)
            cv2.waitKey(1)

            action = self.controller.act(state=prev_state)

            _, reward, done, info = self.env.step(action=action)
            prev_env_state = self.env.ale.getScreenGrayscale()
            prev_state = self._process_state(prev_env_state)
