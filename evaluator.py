import torch

import numpy as np
import time

from pysc2.lib import actions
from pysc2.env import sc2_env
from pysc2.agents import random_agent

import myagent
from models import DQN
from util import hard_update

import pdb
import sys
from absl import flags
FLAGS = flags.FLAGS

Tensor = torch.cuda.DoubleTensor


def make_action(action_id, action_arg):
    return actions.FunctionCall(action_id, action_arg)


def save_checkpoint(state, times):
    torch.save(state, 'trained/checkpoint' + str(times) + '.pth.tar')


def run_loop(env, shared_actor, n_test, max_frames=0):
    model = DQN().cuda()
    hard_update(model, shared_actor)
    total_frames = 0
    total_reward = []
    #pdb.set_trace()
    try:
        for episode in range(n_test):
            #actor.load_state_dict(shared_actor.state_dict())
            episode_reward = 0
            timesteps = env.reset()
            agent = myagent.MyAgent()
            agent.update_model(model)
            action_spec = env.action_spec()
            observation_spec = env.observation_spec()

            agent.setup(observation_spec, action_spec)
            agent.reset()

            while True:
                total_frames += 1
                #actions = agent.step(timesteps[0])
                select_id, select_arg, _, __ = agent.step(timesteps[0], False, False, True, total_frames)

                actions = make_action(select_id, select_arg)
                #if max_frames and total_frames >= max_frames:
                #    return totalReward
                timesteps = env.step([actions])

                _reward = timesteps[0].reward
                episode_reward += _reward
                done = timesteps[0].last()

                if done:
                    total_reward.append(episode_reward)
                    break

    except KeyboardInterrupt:
        return total_reward

    finally:
        #elapsed_time = time.time() - start_time
        #print("Took %.3f seconds for %s steps: %.3f fps" % (elapsed_time, total_frames, total_frames / elapsed_time))
        return total_reward


def evaluate(params, shared_actor, memory):
    argv = FLAGS(sys.argv)
    times = 0
    f = open('./log/distributed_1.txt', 'w')
    f.write('descript here\n\n')

    max_reward = 0
    while True:
        memory_len = memory.len()
        if memory_len < params.batch_size:
            time.sleep(3)
        else:
            times += 1
            start_time = time.time()
            #hard_update(actor, shared_actor)

            with sc2_env.SC2Env(map_name="CollectMineralShards", visualize=False) as env:
                rewards = run_loop(env, shared_actor, params.n_test, 0)

                for i in range(len(rewards)):
                    if rewards[i] >= max_reward:
                        max_reward = rewards[i]
                        env.save_replay("learning_171123/")
                        save_checkpoint({
                            'state_dict': shared_actor.state_dict()
                        }, times)

                if times % 500 == 0:
                    env.save_replay("learning_171123/")
                    save_checkpoint({
                        'state_dict': shared_actor.state_dict()
                    }, times)
                env.close()

            end_time = time.time()
            # result = np.array(result).reshape(-1, 1)
            print('times: {}\tmean_result: {}\tmemory_size: {}\nresult: {}'.format(times, np.mean(rewards), memory_len,rewards))
            f.write('times: {}\tmean_result: {}\tresult: {}\t memory_size: {}\n'.format(times, np.mean(rewards), rewards, memory_len))
            f.write('evaluate time: {}\n'.format(end_time - start_time))
            f.flush()


if __name__ == '__main__':
    argv = FLAGS(sys.argv)
    times = 0
    max_reward = 0
    actor = DQN().cuda()
    agent = myagent.MyAgent(actor)
   # while True:
    result = []
    times += 1
    for episode in range(1):
        with sc2_env.SC2Env(map_name="CollectMineralShards", visualize=False) as env:
            #agent = random_agent.RandomAgent()
            rewards = run_loop(agent, env, 2, 0)
            pdb.set_trace()
            #env.save_replay("eval/")
            env.close()
            print(rewards)

    # result = np.array(result).reshape(-1, 1)
    print('times: {}\tmean_result: {}\tresult: {}'.format(times, np.mean(rewards), rewards))
    sys.exit()
