from .hdqn_models import MetaControllerModel, ControllerModel
import gym
import numpy as np
import torch
import cv2
import gym.spaces
import matplotlib.pyplot as plt


class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.index = 0

        self.states = np.zeros(shape=(size, 84, 84*1), dtype=np.uint8)
        self.next_states = np.zeros(shape=(size, 84, 84*1), dtype=np.uint8)
        self.rewards = np.zeros((size, 1))
        self.done = np.zeros((size, 1))
        self.actions = np.zeros((size, 1), dtype=np.uint8)

    def store(self, state, next_state):
        self.states[self.index] = state
        self.states[self.index] = next_state

        self.index = (self.index+1) % self.size
        print(self.index)


class MetaController:
    def __init__(self):
        self.subgoals = []
        self.subgoals.append(SubGoal(name="middle-ladder", imgsize=84, width=3, height=3, originx=40, originy=48))
        self.subgoals.append(SubGoal(name="top-left-door", imgsize=84, width=4, height=10, originx=13, originy=25))
        self.subgoals.append(SubGoal(name="top-right-door", imgsize=84, width=4, height=10, originx=65, originy=25))
        self.subgoals.append(SubGoal(name="bottom-left-ladder", imgsize=84, width=4, height=10, originx=10, originy=60))
        self.subgoals.append(SubGoal(name="bottom-right-ladder", imgsize=84, width=4, height=10, originx=69, originy=60))
        self.subgoals.append(SubGoal(name="key", imgsize=84, width=3, height=3, originx=6, originy=40))

        self.action_space = gym.spaces.Discrete(n=len(self.subgoals))
        self.model = MetaControllerModel(n_options=self.action_space.n, device='cuda')

    def sample_subgoal(self, state):
        subgoal_id = self.action_space.sample()
        return subgoal_id, self.subgoals[subgoal_id].mask

    def subgoal_validator(self, state, subgoal_id):
        return self.subgoals[subgoal_id].is_goal_reached(frame=state)


class Controller:
    def __init__(self, action_space):
        self.action_space = action_space
        self.model = ControllerModel(n_actions=action_space.n, device='cuda')
        self.buffer = ReplayBuffer(size=int(1e6))

    def act(self, state, subgoal_mask):
        return self.action_space.sample()


class SubGoal:
    def __init__(self, name, imgsize, originx, originy, height, width):
        self.imgsize = imgsize
        self.originx = originx
        self.originy = originy
        self.height = height
        self.width = width
        self.name = name
        self.init_frame_hash = None
        self.mask = self._get_mask()

    def _get_frame_hash(self, frame):
        # calculate a simple sum of all pixels in the goal mask
        goal_viewport = frame[0, self.originy:(self.originy + self.height), self.originx:(self.originx + self.width)]
        return sum(sum(goal_viewport))

    def _get_mask(self):
        goal = np.zeros((self.imgsize, self.imgsize), dtype=np.uint8)

        for row_index in range(0, self.imgsize):
            for col_index in range(0, self.imgsize):
                if self.originy < row_index < self.originy+self.height and self.originx < col_index < self.originx + self.width:
                    goal[row_index][col_index] = 255
        return goal

    def is_goal_reached(self, frame):
        # compare the pixel count within the mask. If it changed then the agent has reached the goal
        if self.init_frame_hash is None:
            # use first frame as reference frame
            self.init_frame_hash = self._get_frame_hash(frame)

        frame_hash = self._get_frame_hash(frame)
        return abs(frame_hash-self.init_frame_hash) > 1


class hDQN:

    def __init__(self):
        self.env = gym.make("MontezumaRevengeNoFrameskip-v4")

        self.meta_controller = MetaController()
        self.controller = Controller(action_space=self.env.action_space)

    def _process_state(self, state):
        state = cv2.resize(state, (84, 84))
        state = state.reshape((1, 84, 84))
        return state

    def run(self, n_steps):

        done = True
        subgoal_reached = False
        prev_env_state, prev_state, subgoal_mask, subgoal = None, None, None, None

        for _ in range(n_steps):

            if done:
                self.env.reset()
                # get the observation from the ale buffer (this is faster than converting the color image to grayscale)
                prev_env_state = self.env.ale.getScreenGrayscale()
                prev_state = self._process_state(prev_env_state)
                subgoal_reached = True
                print("Started new ep")

            if subgoal_reached:
                subgoal, subgoal_mask = self.meta_controller.sample_subgoal(state=prev_state)
                print(f"Selected new subgoal {subgoal}")

            #cv2.imshow("render", prev_env_state)
            #cv2.waitKey(1)

            action = self.controller.act(state=prev_state, subgoal_mask=subgoal_mask)

            _, reward, done, info = self.env.step(action=action)

            result_env_state = self.env.ale.getScreenGrayscale()
            result_state = self._process_state(result_env_state)

            subgoal_reached = self.meta_controller.subgoal_validator(result_state, subgoal_id=subgoal)

            self.controller.buffer.store(state=prev_state, next_state=result_state)

            prev_state = result_state
            prev_env_state = result_env_state
