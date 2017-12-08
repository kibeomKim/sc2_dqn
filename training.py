import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from util import soft_update, hard_update
import time
import copy

import pdb

Tensor = torch.cuda.DoubleTensor
criterion = nn.MSELoss()


def save_checkpoint(state, times):
    torch.save(state, 'checkpoint_training_' + str(times) + '.pth.tar')


def update_policy(batch, dqn, dqn_optimizer, dqn_target, total_frames, params):
    done_batch = np.stack(batch.done)
    done_batch = torch.from_numpy(done_batch).cuda()
    done_batch = done_batch.type(torch.cuda.FloatTensor)

    reward_batch = np.stack(batch.reward)
    reward_batch = torch.from_numpy(reward_batch).cuda()
    reward_batch = reward_batch.type(torch.cuda.FloatTensor)

    action_batch = np.stack(batch.action)
    action_batch = torch.from_numpy(action_batch).cuda()
    action_batch = action_batch.type(torch.cuda.LongTensor)

    state_screen_batch = np.stack(batch.state_screen)
    state_screen_batch = torch.from_numpy(state_screen_batch).cuda()
    state_screen_batch = state_screen_batch.type(torch.cuda.FloatTensor)
    state_screen_batch = Variable(state_screen_batch)

    state_player_batch = np.stack(batch.player)
    state_player_batch = torch.from_numpy(state_player_batch).cuda()
    state_player_batch = state_player_batch.type(torch.cuda.FloatTensor)
    state_player_batch = Variable(state_player_batch)

    state_other_batch = np.stack(batch.other)
    state_other_batch = torch.from_numpy(state_other_batch).cuda()
    state_other_batch = state_other_batch.type(torch.cuda.FloatTensor)
    state_other_batch = Variable(state_other_batch)

    state_batch = [state_screen_batch, state_player_batch, state_other_batch]

    next_state_screen_batch = np.stack(batch.next_state_screen)
    next_state_screen_batch = torch.from_numpy(next_state_screen_batch).cuda()
    next_state_screen_batch = next_state_screen_batch.type(torch.cuda.FloatTensor)
    next_state_screen_batch = Variable(next_state_screen_batch, volatile=True)

    next_player_batch = np.stack(batch.next_player)
    next_player_batch = torch.from_numpy(next_player_batch).cuda()
    next_player_batch = next_player_batch.type(torch.cuda.FloatTensor)
    next_player_batch = Variable(next_player_batch, volatile=True)

    next_other_batch = np.stack(batch.next_other)
    next_other_batch = torch.from_numpy(next_other_batch).cuda()
    next_other_batch = next_other_batch.type(torch.cuda.FloatTensor)
    next_other_batch = Variable(next_other_batch, volatile=True)

    next_state_batch = [next_state_screen_batch, next_player_batch, next_other_batch]

    # pdb.set_trace()
    # Prepare for the target q batch
    if total_frames % 1800 == 0:
        #pdb.set_trace()
        pass

    q_pred = dqn(state_batch)
    q_values = q_pred.gather(1, Variable(action_batch))

    target_pred = dqn_target(next_state_batch)
    target_pred.volatile = False
    target_pred = torch.unsqueeze(target_pred.max(1)[0], dim=1)
    target_values = Variable(reward_batch) + target_pred * params.gamma * Variable(done_batch)

    loss = F.smooth_l1_loss(q_values, target_values)

    dqn_optimizer.zero_grad()
    loss.backward()
    # for param in dqn.parameters():
    #    param.grad.data.clamp_(-1, 1)
    dqn_optimizer.step()

    if total_frames % params.update_interval == 0 :
        dqn_target = copy.deepcopy(dqn)


def run_loop(shared_memory, params, shared_model, target_model, optimizer):
    times = 0
    f = open('./log/distributed_1_log.txt', 'w')
    f.write('learning_time\n')
    while True:
        if shared_memory.len() < params.batch_size:
            time.sleep(2)
        else:
            time.sleep(0.2)
            for i in range(params.episode):
                start_time = time.time()
                times += 1
                memory_len = shared_memory.len()

                batch = shared_memory.sample(params.batch_size)
                update_policy(batch, shared_model, optimizer, target_model, times, params)
                #update_policy_right(batch, actor, critic, actor_target, critic_target, actor_optimizer, critic_optimizer, params)

                end_time = time.time()
                f.write('times: {}\ttime: {}\tmemory: {}\n'.format(times, end_time - start_time ,memory_len))

                '''
                if times % 9 == 0:
                    save_checkpoint({
                        'state_dict': actor.state_dict()
                    }, times)
                '''
                '''
                if times % 20 == 0 :
                    print('memory len: ' + str(memory.len()))
                    result, mean_result, result_step = evaluate(params, actor, times, env)
                    f.write('times: {}\t result: {}\tmean_result: {}\tstep: {}\t memory_size: {}'.format(times, result,
                                                                                                         mean_result,
                                                                                                         result_step,
                                                                                                         memory.len()))
                '''
