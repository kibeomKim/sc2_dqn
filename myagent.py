import torch
from torch.autograd import Variable

from pysc2.agents import base_agent
from pysc2.lib import actions
from s2clientprotocol import ui_pb2 as sc_ui
from models import DQN
import random
import numpy as np
import random
import math

import pdb

# torch.set_default_tensor_type('torch.cuda.IntTensor')
Tensor = torch.cuda.DoubleTensor

'''
======================================================
self.step : 24
obs type : <class 'pysc2.env.environment.TimeStep'>
------------------------------------------------------
obs.step_type : <enum 'StepType'>
StepType.MID    [FIRST, MID, LAST]
------------------------------------------------------
obs.reward : <class 'int'>
0
------------------------------------------------------
obs.discount : <class 'float'>
1.0
------------------------------------------------------
obs.observation : <class 'dict'> - {str : numpy.ndarray}
obs.observation['build_queue'] : (n, 7)
    ..['build_queue'][i][j] : same as single_select
obs.observation['game_loop'] : (1,)
obs.observation['cargo_slots_available'] : (1,)
obs.observation['player'] : (11,)
    ..['player'][0] : player_id
    ..['player'][1] : mineral
    ..['player'][2] : vespine
    ..['player'][3] : food used
    ..['player'][4] : food cap
    ..['player'][5] : food used by army
    ..['player'][6] : food used by workers
    ..['player'][7] : idle worker count
    ..['player'][8] : army count
    ..['player'][9] : warp gate count
    ..['player'][10] : larva count
obs.observation['available_actions'] : (n)
    ..['available_actions'][i] : available action id
obs.observation['minimap'] : (7, 64, 64)
    ..['minimap'][0] : height_map
    ..['minimap'][1] : visibility
    ..['minimap'][2] : creep
    ..['minimap'][3] : camera
    ..['minimap'][4] : player_id
    ..['minimap'][5] : player_relative              < [0,4] < [background, self, ally, neutral, enemy]
    ..['minimap'][6] : selected                     < 0 for not selected, 1 for selected
obs.observation['cargo'] : (n, 7) - n is the number of all units in a transport
    ..['cargo'][i][j] :  same as single_select[0][j]
obs.observation['multi_select'] : (n, 7)
    ..['multi_select'][i][j] : same as single_select[0][j]
        ->  Not Exist with single_select
           when single_select, multi_select=[]
           when multi_select, single_select = [[0,0,0,0,0,0,0]]
obs.observation['score_cumulative'] : (13,)
obs.observation['control_groups'] : (10, 2)
    ..['control_groups'][i][0] : i'th unit leader type
    ..['control_groups'][i][1] : count
obs.observation['single_select'] : (1, 7)
    ..['single_select'][0][0] : unit_type
    ..['single_select'][0][1] : player_relative     < [0,4] < [background, self, ally, neutral, enemy]
    ..['single_select'][0][2] : health
    ..['single_select'][0][3] : shields
    ..['single_select'][0][4] : energy
    ..['single_select'][0][5] : transport slot
    ..['single_select'][0][6] : build progress as percentage
obs.observation['screen'] : (13, 84, 84) or (13, 64, 64)
    ..['screen'][0] : height_map
    ..['screen'][1] : visibility
    ..['screen'][2] : creep
    ..['screen'][3] : power                         < protoss power
    ..['screen'][4] : player_id                     < cristal shards=16, me = 1
    ..['screen'][5] : player_relative               < [0,4] < [background, self, ally, neutral, enemy]
    ..['screen'][6] : unit_type                     < cristal shards=1680, marine=48
    ..['screen'][7] : selected                      < 0 for not selected, 1 for selected    
    ..['screen'][8] : hit_points
    ..['screen'][9] : energy
    ..['screen'][10] : shields
    ..['screen'][11] : unit_density
    ..['screen'][12] : unit_density_aa
======================================================
'''

_no_op = actions.FUNCTIONS.no_op.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id
Single_Select = sc_ui.ActionMultiPanel.SingleSelect
Control_Group_Set = sc_ui.ActionControlGroup.Set
Control_Group_Select = 0


def intToCoordinate(num, size=63):
    if size != 63:
        num = num * size * size // 4096
    y = num // size
    x = num - size * y
    return [x, y]


def coordinateToInt(x, y, size=63):
    return x + size * y


def screen_process(obs):
    # np.set_printoptions(threshold=np.inf)
    # data = np.stack((obs.observation["screen"][4]/16, obs.observation["screen"][5]/4, obs.observation["screen"][6]/1680, obs.observation["screen"][7]))
    # data = data.astype(np.float)
    # print(obs.observation["screen"][4])

    # pdb.set_trace()
    new_screen = (obs.observation["screen"][5] == 3).astype(np.int)
    new_screen = np.expand_dims(new_screen, axis=0)
    player_y, player_x = (obs.observation["screen"][7] == 1).nonzero()  # my player
    if len(player_y) == 0:
        player_y = 0
    else:
        player_y = int(player_y.mean())
    if len(player_x) == 0:
        player_x = 0
    else:
        player_x = int(player_x.mean())
    player = np.array([player_x, player_y])

    '''
    other_player_y, other_player_x = (obs.observation["screen"][5] == 1).nonzero()
    player1_x = int((other_player_x[:4]).mean())
    player1_y = int((other_player_y[:4]).mean())
    player2_x = int((other_player_x[4:]).mean())
    player2_y = int((other_player_y[4:]).mean())

    if player_x == player1_x and player_y == player1_y:
        other_player_x = player2_x
        other_player_y = player2_y
    else:
        other_player_x = player1_x
        other_player_y = player1_y
    '''
    other_player_x = 0.
    other_player_y = 0.
    other_player = np.array([other_player_x, other_player_y])
    return new_screen, player, other_player


def action_process(action):
    # new_action = action*63
    new_action = np.clip(action, 0., 63.)  # 0 to 63
    return new_action


def get_angle(to, fr):
    dx = abs(fr[0] - to[0])
    dy = abs(fr[1] - to[1])
    rad = math.atan2(dy, dx)
    # degree = (rad * 180)/math.pi
    return rad


def get_point(point, degree):
    if degree == 0:
        point[0] += 24
        pass
    elif degree == 1:
        point[0] += 18
        point[1] += 18
        pass
    elif degree == 2:
        point[1] += 24
        pass
    elif degree == 3:
        point[0] -= 18
        point[1] += 18
        pass
    elif degree == 4:
        point[0] -= 24
        pass
    elif degree == 5:
        point[0] -= 18
        point[1] -= 18
        pass
    elif degree == 6:
        point[1] -= 24
        pass
    elif degree == 7:
        point[0] += 18
        point[1] -= 18
        pass

    return action_process(point)


def add_noise(action, random_process, epsilon):
    action += max(epsilon, 0) * random_process.sample()
    return action


class MyAgent(base_agent.BaseAgent):
    def __init__(self, model=None):
        super(MyAgent, self).__init__()
        self.odd = 1
        self.control_num = 0
        self.unit_id = 0
        self.model = DQN().cuda()
        self.t1 = {}
        self.t2 = {}
        self.state_multi = []
        self.epsilon = 1.0
        self.ou_theta = 0.15
        self.ou_sigma = 0.3
        self.ou_mu = 0.0
        self.actions = 1

    def setup(self, obs_spec, action_spec):
        super(MyAgent, self).setup(obs_spec, action_spec)

    def update_model(self, new_model):
        self.model.load_state_dict(new_model.state_dict())

    def reset(self):
        self.odd = 1
        self.control_num = 0
        self.unit_id = 0

    def step(self, obs, random_action=False, noise=False, eval=False, total_frames=0):
        # obs, state = obss
        super(MyAgent, self).step(obs)
        # pdb.set_trace()
        All_Unit_Num = obs.observation['player'][3]
        # state_screen = Variable(Tensor(np.array([state_screen]).astype(np.float32)).cuda())
        # function_id = np.random.choice(obs.observation["available_actions"])
        state_data = [0]
        degree = 0
        select_id = 0
        select_arg = 0
        if self.odd == 1:
            select_id = _SELECT_ARMY
            select_arg = [_SELECT_ALL]
            self.odd = 2

        elif self.odd == 2:
            select_id = _SELECT_UNIT
            select_arg = [[Single_Select], [self.unit_id]]
            # if self.flag is True:
            # pdb.set_trace()
            self.state_multi = np.concatenate(
                (obs.observation['multi_select'][0][:3], obs.observation['multi_select'][1][:3]))
            self.state_multi = self.state_multi.astype(np.float)

            self.odd = 5
            # self.unit_id = 0
            # action
            '''
            else:
                self.odd = 1    #3
            '''
            self.unit_id += 1
            '''
        elif self.odd == 3:
            select_id = _SELECT_CONTROL_GROUP
            select_arg = [[Control_Group_Set], [self.control_num]]
            self.control_num += 1
            if self.control_num == All_Unit_Num:
                self.odd = 5
                self.control_num = 0
            else:
                self.odd = 1

        elif self.odd == 4:
            select_id = _SELECT_CONTROL_GROUP
            select_arg = [[Control_Group_Select], [self.control_num]]
            self.control_num += 1

            self.odd = 5
            '''
        elif self.odd == 5:
            # args = [[np.random.randint(0, size) for size in arg.sizes] for arg in self.action_spec.functions[_MOVE_SCREEN].args]
            screen_data, player_raw, other_player = screen_process(obs)

            state_screen = torch.from_numpy(screen_data)
            state_screen_data = torch.unsqueeze(state_screen, 0)  # state
            state_screen = Variable(state_screen_data).cuda()
            state_screen = state_screen.type(torch.cuda.FloatTensor)

            player = player_raw / 63
            player_data = torch.from_numpy(player)
            player_data = torch.unsqueeze(player_data, 0)
            player_data = Variable(player_data).cuda()
            player_data = player_data.type(torch.cuda.FloatTensor)

            other_player = other_player / 63
            other_player_data = torch.from_numpy(other_player)
            other_player_data = torch.unsqueeze(other_player_data, 0)
            other_player_data = Variable(other_player_data).cuda()
            other_player_data = other_player_data.type(torch.cuda.FloatTensor)

            state = [state_screen, player_data, other_player_data]
            state_data = [screen_data, player, other_player]

            if random_action is True:
                # _args = [[np.random.randint(0, size) for size in arg.sizes] for arg in self.action_spec.functions[_MOVE_SCREEN].args]
                degree = random.randrange(0, 8)
                point = get_point(player_raw, degree)
                degree = np.array([degree])
                x = point[0]
                y = point[1]
                _args = [[0], [x, y]]
            else:
                if eval is True:
                    self.model.eval()
                else:
                    self.model.train()
                z = self.model(state)

                degree = z.data.cpu().max(1)[1]
                degree = degree.numpy()

                point = get_point(player_raw, degree[0])
                x = point[0]
                y = point[1]

                _args = [[0], [x, y]]

            # print(args)
            # args = [[np.random.randint(0, size) for size in arg.sizes] for arg in self.action_spec.functions[_MOVE_SCREEN].args]
            select_id = _MOVE_SCREEN
            select_arg = _args
            self.odd = 1
            if self.unit_id == All_Unit_Num:
                self.control_num = 0
                self.unit_id = 0

        # print("step :", self.steps, "action id :", select_id, "arg :", select_arg, "odd:", self.odd)
        # return actions.FunctionCall(select_id, select_arg)
        return select_id, select_arg, degree, state_data


'''
173   Attributes:
174     0  screen: A point on the screen.
175     1  minimap: A point on the minimap.
176     2  screen2: The second point for a rectangle. This is needed so that no
177          function takes the same type twice.
178     3  queued: Whether the action should be done now or later.                 size<2
179     4  control_group_act: What to do with the control group.                   size<5
180     5  control_group_id: Which control group to do it with.                    size<10
181     6  select_point_act: What to do with the unit at the point.                size<4
182     7  select_add: Whether to add the unit to the selection or replace it.     size<2
183     8  select_unit_act: What to do when selecting a unit by id.                size<4
184     9  select_unit_id: Which unit to select by id.                             size<500
185     10 select_worker: What to do when selecting a worker.                      size<4
186     11 build_queue_id: Which build queue index to target.                      size<10
187     12 unload_id: Which unit to target in a transport/nydus/command center.    size<500
'''
