import os
import torch.multiprocessing as mp
import torch.optim as optim

from play import running
from models import DQN
from replayMemory import ReplayMemory
from training import run_loop
from util import hard_update
from multiprocessing import Manager
from multiprocessing.managers import BaseManager
from evaluator import evaluate

from absl import flags

import pdb


class Params():
    def __init__(self):
        self.n_process = 4
        self.episode = 100000
        self.batch_size = 1024  # need to modify
        self.gamma = 0.9
        self.warmup = 200       # need to modify
        self.n_test = 1
        self.depsilon = 1.0/10000  # need to modify
        self.update_interval = 500

#eval(), load_state_dict(model.state_dict()), train()


if __name__ == '__main__':
    params = Params()
    mp.set_start_method('spawn')

    shared_model = DQN().cuda()
    shared_model.share_memory()

    optimizer = optim.Adam(shared_model.parameters(), lr=0.001)

    target_model = DQN().cuda()
    hard_update(target_model, shared_model)
    target_model.share_memory()

    BaseManager.register('ReplayMemory', ReplayMemory)
    manager = BaseManager()
    manager.start()
    shared_memory = manager.ReplayMemory()

    #memory = ReplayMemory()

    processes = []

    p = mp.Process(target=evaluate, args=(params, shared_model, shared_memory, ))
    p.start()
    processes.append(p)

    p = mp.Process(target=run_loop, args=(shared_memory, params, shared_model, target_model, optimizer, ))
    p.start()
    processes.append(p)
    for rank in range(params.n_process):
        p = mp.Process(target=running, args=(rank, params, shared_model, shared_memory, ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()