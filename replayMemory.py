import torch

import numpy as np
from random import sample
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state_screen', 'player', 'other', 'action', 'reward', 'next_state_screen', 'next_player', 'next_other', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)
        self.position = 0

    def put(self, state, action, reward, next_state, done):
        reward = np.asarray([reward]).astype(np.float)
        done = np.asarray([done]).astype(np.float)
        #action = np.asarray([action]).astype(np.float)
        state_screen, player, other = state
        next_state_screen, next_player, next_other = next_state

        self.memory.append(Transition(state_screen, player, other, action, reward, next_state_screen, next_player, next_other, done))
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions = sample(self.memory, batch_size)
        return Transition(*(zip(*transitions)))

    def __len__(self):
        return len(self.memory)

    def len(self):
        return len(self.memory)