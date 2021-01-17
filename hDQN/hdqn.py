from .hdqn_models import MetaControllerModel, ControllerModel
import gym
import numpy as np
import torch
import cv2
import gym.spaces
from torch.optim import Adam
from torch.nn.functional import smooth_l1_loss
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.index = 0
        self.size = 0

        self.states = np.zeros(shape=(self.capacity, 84, 84*2), dtype=np.uint8)
        self.next_states = np.zeros(shape=(self.capacity, 84, 84*2), dtype=np.uint8)
        self.rewards = np.zeros((self.capacity, 1))
        self.non_terminal = np.zeros((self.capacity, 1))
        self.actions = np.zeros((self.capacity, 1), dtype=np.uint8)

    def store(self, state, next_state, action, reward, done):
        self.states[self.index] = state
        self.next_states[self.index] = next_state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.non_terminal[self.index] = not done

        if self.index > self.size:
            self.size = self.index

        self.index = (self.index+1) % self.capacity

    def get_samples(self, size):
        idx = np.random.default_rng().choice(self.size, size=size, replace=False)

        states = self.states[idx]
        next_states = self.next_states[idx]
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        non_terminal = self.non_terminal[idx]

        return states, next_states, actions, rewards, non_terminal


class MetaController:
    def __init__(self):
        self.subgoals = []
        self.subgoals.append(SubGoal(name="middle-ladder", imgsize=84, width=3, height=3, originx=40, originy=48))
        #self.subgoals.append(SubGoal(name="top-left-door", imgsize=84, width=4, height=10, originx=13, originy=25))
        #self.subgoals.append(SubGoal(name="top-right-door", imgsize=84, width=4, height=10, originx=65, originy=25))
        #self.subgoals.append(SubGoal(name="bottom-left-ladder", imgsize=84, width=4, height=10, originx=10, originy=60))
        self.subgoals.append(SubGoal(name="bottom-right-ladder", imgsize=84, width=4, height=10, originx=69, originy=60))
        #self.subgoals.append(SubGoal(name="key", imgsize=84, width=3, height=3, originx=6, originy=40))

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
        self.device = "cuda"

        model_config = {"n_actions": action_space.n, "device": self.device}
        self.model = ControllerModel(**model_config)
        self.target_model = ControllerModel(**model_config)
        self.sync_models()
        self.optimizer = Adam(params=self.model.parameters(), lr=0.0000625)
        self.criterion = smooth_l1_loss
        self.min_eps = 0.01
        self.eps_decay_step = (1-self.min_eps)/int(1e6)
        self.eps = 1

        self.buffer = ReplayBuffer(capacity=int(1e6))

    def sync_models(self):
        self.target_model.load_state_dict(self.model.state_dict())

    @staticmethod
    def process_state(state, subgoal_mask):
        state = np.concatenate((state, subgoal_mask), axis=2)
        return state

    def greedy_action(self, state):
        with torch.no_grad():
            values = self.model(torch.tensor([state/255.0], dtype=torch.float, device=self.device))
            action = values.argmax().item()
            return action

    def save(self):
        torch.save(self.model, "controller.pth")

    def act(self, state):

        if np.random.rand() < self.eps:
            action = self.action_space.sample()
        else:
            action = self.greedy_action(state)

        if self.eps > self.min_eps:
            self.eps -= self.eps_decay_step

        return action

    def update(self):
        states, next_states, actions, rewards, non_terminal = self.buffer.get_samples(size=32)

        states = torch.tensor(states/255.0, device=self.device, dtype=torch.float).unsqueeze(1)
        next_states = torch.tensor(next_states / 255.0, device=self.device, dtype=torch.float).unsqueeze(1)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)
        non_terminal = torch.tensor(non_terminal, dtype=torch.long, device=self.device)

        # get the current Q(s_t, a_t)
        values = self.model(states)
        qa_values = torch.gather(values, 1, actions)

        with torch.no_grad():
            future_value = self.target_model(next_states)
            future_value, _ = future_value.max(dim=1)
            future_value = future_value.unsqueeze(1)

        # calculate the new expected action-values
        new_qa = rewards + non_terminal*0.99*future_value

        # calculate the loss
        loss = self.criterion(input=qa_values, target=new_qa)

        # update the network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


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

            self.env.render()

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

            if step_id % int(80e3) == 0:
                print("syncing models")
                self.controller.sync_models()
                self.controller.save()

            if step_id > int(10e3):
                if step_id % 4 == 0:
                    self.controller.update()

            prev_state = result_state
