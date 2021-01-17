from .hdqn_models import MetaControllerModel, ControllerModel
import gym
import numpy as np
import torch
import cv2
import gym.spaces
from collections import deque
from .controller import Controller
from .meta_controller import MetaController


class hDQN:
    def __init__(self):
        self.env = gym.make("MontezumaRevengeNoFrameskip-v4")

        # only use the relevant action for the game
        self.actions = [1, 2, 3, 4, 5]

        self.meta_controller = MetaController()
        self.controller = Controller(action_space=gym.spaces.Discrete(len(self.actions)))

    def _process_state(self, state):
        state = cv2.resize(state, (84, 84))
        state = state.reshape((1, 84, 84))
        return state

    def run(self, n_steps):
        done = True
        subgoal_reached = False
        lives = 6
        prev_env_state, prev_state, subgoal_mask, subgoal = None, None, None, None

        subgoal_history = deque(maxlen=100)

        for step_id in range(n_steps):
            if done:
                self.env.reset()
                # get the observation from the ale buffer (this is faster than converting the color image to grayscale)
                prev_env_state = self.env.ale.getScreenGrayscale()
                prev_state = self._process_state(prev_env_state)

                if not subgoal_reached:
                    subgoal_history.append(0)

                subgoal, subgoal_mask = self.meta_controller.sample_subgoal(state=prev_state)

                print(f"steps: {step_id} success: {np.round(np.average(subgoal_history),2)} eps: {np.round(self.controller.eps, 2)}")

            if subgoal_reached:
                subgoal, subgoal_mask = self.meta_controller.sample_subgoal(state=prev_state)

            prev_controller_state = self.controller.process_state(prev_state, subgoal_mask)

            action = self.controller.act(state=prev_controller_state)
            _, reward, done, info = self.env.step(action=self.actions[action])

            result_env_state = self.env.ale.getScreenGrayscale()
            result_state = self._process_state(result_env_state)
            result_controller_state = self.controller.process_state(result_state, subgoal_mask)

            subgoal_reached = self.meta_controller.subgoal_validator(result_state, subgoal_id=subgoal)

            if subgoal_reached:
                subgoal_history.append(1)

            controller_done = subgoal_reached

            if info['ale.lives'] < lives:
                lives = info['ale.lives']
                controller_done = True

            self.controller.buffer.store(state=prev_controller_state, next_state=result_controller_state, action=action, reward=subgoal_reached, done=controller_done)

            if step_id % int(8e3) == 0:
                self.controller.sync_models()

            if step_id % int(100e3) == 0:
                self.controller.save()

            if step_id > int(100e3) == 0:
                if step_id % 4 == 0:
                    self.controller.update()

            prev_state = result_state
