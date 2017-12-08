import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import pdb


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class share_gradient():
    def __init__(self, model):
        self.grads = {}
        for name, p in model.named_parameters():
            self.grads[name + '_grad'] = torch.zeros(p.size()).share_memory_()

    def add_gradient(self, model):
        for name, p in model.named_parameters():
            self.grads[name + '_grad'] += p.grad.data

    def reset(self):
        for name, grad in self.grads.items():
            self.grads[name].fill_(0)


class DQN(nn.Module):
    def __init__(self, output=8, hidden=512, init_w=3e-3):
        super(DQN, self).__init__()
        self.conv1_screen = nn.Conv2d(1, 8, kernel_size=8, stride=4, padding=0)
        self.conv2_screen = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=0)
        self.conv3_screen = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)  # 256

        self.add_other = nn.Linear(256 + 2, hidden)

        self.affine = nn.Linear(256 + 2, hidden)
        self.affine1 = nn.Linear(hidden, hidden)
        self.value_head = nn.Linear(hidden, output)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.affine.weight.data = fanin_init(self.affine.weight.data.size())
        self.affine1.weight.data = fanin_init(self.affine1.weight.data.size())

    def forward(self, state):

        screen, player, other = state

        x = F.relu(self.conv1_screen(screen))
        x = F.relu(self.conv2_screen(x))
        x = F.relu(self.conv3_screen(x))

        x = x.view(x.size(0), -1)
        xy = torch.cat([x, player], dim=1)

        xy = F.leaky_relu(self.affine(xy))
        xy = F.leaky_relu(self.affine1(xy))

        return self.value_head(xy)