from .hdqn_models import ControllerModel
import numpy as np
import torch
from torch.optim import Adam
from torch.nn.functional import smooth_l1_loss
from .replay_buffer import ReplayBuffer


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