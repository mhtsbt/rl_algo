from .hdqn_models import MetaControllerModel
import numpy as np
import torch
from torch.optim import Adam
from torch.nn.functional import smooth_l1_loss
from .replay_buffer import ReplayBuffer
import gym.spaces


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
        goal = np.zeros((1, self.imgsize, self.imgsize), dtype=np.uint8)

        for row_index in range(0, self.imgsize):
            for col_index in range(0, self.imgsize):
                if self.originy < row_index < self.originy+self.height and self.originx < col_index < self.originx + self.width:
                    goal[0][row_index][col_index] = 255
        return goal

    def is_goal_reached(self, frame):
        # compare the pixel count within the mask. If it changed then the agent has reached the goal
        if self.init_frame_hash is None:
            # use first frame as reference frame
            self.init_frame_hash = self._get_frame_hash(frame)

        frame_hash = self._get_frame_hash(frame)
        return abs(frame_hash-self.init_frame_hash) > 1


class MetaController:
    def __init__(self):
        self.subgoals = []
        self.subgoals.append(SubGoal(name="middle-ladder", imgsize=84, width=3, height=3, originx=40, originy=48))
        #self.subgoals.append(SubGoal(name="top-left-door", imgsize=84, width=4, height=10, originx=13, originy=25))
        #self.subgoals.append(SubGoal(name="top-right-door", imgsize=84, width=4, height=10, originx=65, originy=25))
        #self.subgoals.append(SubGoal(name="bottom-left-ladder", imgsize=84, width=4, height=10, originx=10, originy=60))
        #self.subgoals.append(SubGoal(name="bottom-right-ladder", imgsize=84, width=4, height=10, originx=69, originy=60))
        #self.subgoals.append(SubGoal(name="key", imgsize=84, width=3, height=3, originx=6, originy=40))

        self.action_space = gym.spaces.Discrete(n=len(self.subgoals))
        self.model = MetaControllerModel(n_options=self.action_space.n, device='cuda')

    def sample_subgoal(self, state):
        subgoal_id = self.action_space.sample()
        return subgoal_id, self.subgoals[subgoal_id].mask

    def subgoal_validator(self, state, subgoal_id):
        return self.subgoals[subgoal_id].is_goal_reached(frame=state)

