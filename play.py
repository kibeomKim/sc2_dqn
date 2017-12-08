import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.agents import random_agent

import myagent
from models import DQN
import replayMemory
from util import soft_update, hard_update

import numpy as np
import random
import math
import copy
import sys
import time
from absl import flags
FLAGS = flags.FLAGS

import pdb

Tensor = torch.cuda.DoubleTensor


def make_action(action_id, action_arg):
    return actions.FunctionCall(action_id, action_arg)


def run_loop(env, shared_model, shared_memory, n_episode, max_frames=0):
    total_frames = 0
    totalReward = 0
    #start_time = time.time()
    agent = myagent.MyAgent()
    action_spec = env.action_spec()
    observation_spec = env.observation_spec()
    agent.setup(observation_spec, action_spec)
    random_action = True
    noise = True
    add_reward = 0

    f = open('./log/distributed_1.txt', 'w')
    f.write('descript here\n\n')
    e = 1.0
    try:
        for episode in range(n_episode):
            agent.reset()  # what is this?
            agent.update_model(shared_model)
            totalReward = 0
            timesteps = env.reset()
            game_frames = 0
            e -= 1./10000

            while True:
                total_frames += 1
                game_frames += 1
                if total_frames >= 300 and np.random.rand(1) > e:
                    random_action = False
                else:
                    random_action = True

                select_id, select_arg, _z, _state = agent.step(timesteps[0], random_action, noise, False, total_frames)
                actions = make_action(select_id, select_arg)

                timesteps = env.step([actions])

                _reward = timesteps[0].reward
                done = timesteps[0].last()
                totalReward += _reward

                if done:
                    terminal = 0
                else:
                    terminal = 1

                if game_frames % 3 == 0:
                    if game_frames == 3:
                        state = _state
                        next_state = state
                        z = _z
                        reward = _reward
                    else:
                        state = next_state
                        next_state = _state
                        reward += add_reward
                        shared_memory.put(state, z, reward, next_state, terminal)
                        z = _z
                        reward = _reward
                        add_reward = 0
                else:
                    if _reward != 0:
                        add_reward += _reward

                if done:
                    state = next_state
                    reward = _reward + add_reward
                    shared_memory.put(state, z, reward, next_state, terminal)
                    break

    except KeyboardInterrupt:
        return totalReward

    finally:
        #elapsed_time = time.time() - start_time
        #print("Took %.3f seconds for %s steps: %.3f fps" % (elapsed_time, total_frames, total_frames / elapsed_time))
        return totalReward


def running(rank, params, shared_model, shared_memory):
    argv = FLAGS(sys.argv)
    with sc2_env.SC2Env(map_name="CollectMineralShards", visualize=False) as env:
        run_result = run_loop(env, shared_model, shared_memory, params.episode)
        #print(run_result)
        env.close()


def running_test():
    actor = DQN().cuda()
    #hard_update(actor, shared_actor)
    memory = replayMemory.ReplayMemory()
    total_frames = 0
    with sc2_env.SC2Env(map_name="CollectMineralShards", visualize=False) as env:
        agent = myagent.MyAgent(actor)
        #agent = random_agent.RandomAgent()
        run_result = run_loop(env, actor, memory, 999)
        print(run_result)
        env.close()


def main(unused_argv):
    running_test()


if __name__ == "__main__":
    argv = FLAGS(sys.argv)
    sys.exit(main(argv))