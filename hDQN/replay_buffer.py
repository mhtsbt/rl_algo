import numpy as np


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
