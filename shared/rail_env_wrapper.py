
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator, complex_rail_generator 
from flatland.envs.schedule_generators import sparse_schedule_generator, complex_schedule_generator
from flatland.envs.malfunction_generators import malfunction_from_params
import random

import sys
import warnings
from typing import Callable, Tuple, Optional, Dict, List

import msgpack
import numpy as np

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_utils import get_direction, mirror, direction_to_point
from flatland.core.grid.grid_utils import Vec2dOperations as Vec2d
from flatland.core.grid.grid_utils import distance_on_rail, IntVector2DArray, IntVector2D, \
    Vec2dOperations
from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.grid4_generators_utils import connect_rail_in_grid_map, connect_straight_line_in_grid_map, \
    fix_inner_nodes, align_cell_to_city

RailGeneratorProduct = Tuple[GridTransitionMap, Optional[Dict]]
RailGenerator = Callable[[int, int, int, int], RailGeneratorProduct]

import warnings
from typing import Tuple, List, Callable, Mapping, Optional, Any

import msgpack
import numpy as np

from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.schedule_utils import Schedule

AgentPosition = Tuple[int, int]
ScheduleGenerator = Callable[[GridTransitionMap, int, Optional[Any], Optional[int]], Schedule]


def comm_schedule_generator() -> ScheduleGenerator:
   
    def generator(rail: GridTransitionMap, num_agents: int, hints: Any = None, num_resets: int = 0) -> Schedule:
        start_goal = hints['start_goal']
        start_dir = hints['start_dir']
        agents_position = [sg[0] for sg in start_goal[:num_agents]]
        agents_target = [sg[1] for sg in start_goal[:num_agents]]
        agents_direction = start_dir[:num_agents]
        speeds = [1.0] * len(agents_position)

        stochastic_data = {
                'prop_malfunction': 0.0,  # Percentage of defective agents
                'malfunction_rate': 0,  # Rate of malfunction occurence
                'min_duration': 0,  # Minimal duration of malfunction
                'max_duration': 0  # Max duration of malfunction
        }

        return Schedule(agent_positions=agents_position, agent_directions=agents_direction,
                        agent_targets=agents_target, agent_speeds=speeds, agent_malfunction_rates=malfunction_from_params(stochastic_data))

    return generator


def comm_rail_generator() -> RailGenerator:
    def generator(width, height, num_agents, num_resets=0):
        grid_map = GridTransitionMap(width=width, height=height, transitions=RailEnvTransitions())
        rail_array = grid_map.grid
        rail_array.fill(0)

        station1 = np.array([9, 2])
        station2 = np.array([9, 18])

        wait_out_1 = np.array([9, 4])
        wait_point_1_l = np.array([7, 6])
        wait_point_1_r = np.array([7, 8])
        wait_in_1 = np.array([9, 10])

        wait_out_2 = np.array([9, 16])
        wait_point_2_r = np.array([11, 14])
        wait_point_2_l = np.array([11, 12])
        wait_in_2 = np.array([9, 10])

        offset_x = np.array([0, 1])
        offset_y = np.array([1, 0])

        rail_trans = grid_map.transitions

        def connect(p1, p2,flip_start=False, flip_end = False):
            p1 = tuple(p1)
            p2 = tuple(p2)
            connect_rail_in_grid_map(grid_map, p1, p2, rail_trans, Vec2d.get_chebyshev_distance,
                                                flip_start_node_trans=flip_start, flip_end_node_trans=flip_end,
                                                respect_transition_validity=True, forbidden_cells=None)

        def connect_straight(p1, p2):
            p1 = tuple(p1)
            p2 = tuple(p2)

            connect_straight_line_in_grid_map(grid_map, p1, p2, rail_trans)



        #           / w1 \
        # s1 --- out1 --- in1 --- in2 --- out2 --- s2
        #                           \ w2 /


        connect(station1 - offset_x, wait_out_1 + offset_x)
        connect(wait_out_1, wait_in_1)
        connect(wait_in_1 - offset_x, wait_in_2 + offset_x)  
        connect(station2 + offset_x, wait_out_2 - offset_x)
        connect(wait_out_2, wait_in_2  - offset_x)

        connect(wait_point_1_l, wait_out_1)
        connect_straight(wait_point_1_r, wait_point_1_l)
        connect(wait_point_1_r, wait_in_1)

        connect(wait_point_2_l, wait_out_2)
        connect_straight(wait_point_2_r, wait_point_2_l)
        connect(wait_point_2_r, wait_in_2)
       

        station1 = tuple(station1)
        station2 = tuple(station2)

        start_goal = [[station1, station2],[station2, station1]]
        start_dir = [1,3]

        print(grid_map.grid)

        return grid_map, {'agents_hints': {
            'start_goal': start_goal,
            'start_dir': start_dir
        }}

    return generator



class RailEnvWrapper():
    initial_step_penalty = 0
    global_reward = 0
    
    def __init__(self, observation_builder, width=12, height=12, num_agents=2):
        self.num_agents = num_agents


        self.schedule_gen = comm_schedule_generator()

        '''
        self.schedule_gen = sparse_schedule_generator({   
                    1.: 1.0,       # Fast passenger train
                    1. / 2.: 0,  # Fast freight train
                    1. / 3.: 0,  # Slow commuter train
                    1. / 4.: 0
                })
        '''

        self.stochastic_data = {
                'prop_malfunction': 0.0,  # Percentage of defective agents
                'malfunction_rate': 0,  # Rate of malfunction occurence
                'min_duration': 0,  # Minimal duration of malfunction
                'max_duration': 0  # Max duration of malfunction
        }
        

        self.done_last_step = {}
        self.observation_builder = observation_builder
        self.dist = {}

        self.num_of_done_agents = 0
        self.episode_step_count = 0
        self.max_steps = 40

    def step(self, actions):
        next_obs, rewards, done, info = self.env.step(actions)
        self.done_last_step = dict(done)
        self.episode_step_count += 1
        return next_obs, rewards, done, info
    
    
    def reset(self):
        for i in range(self.num_agents):
            self.done_last_step[i] = False
            self.dist[i] = 100
        
        obs,info = self.env.reset(activate_agents=True)
        
        # Obs-shape must be equal to num of agents, otherwise, the level couldn't be generated orderly
        while self.env.num_agents != self.num_agents:
            obs,info = self.env.reset(activate_agents=True)

        self.num_of_done_agents = 0
        self.env.step_penalty = self.initial_step_penalty
        self.env.global_reward = self.global_reward
        self.episode_step_count = 0
        return obs,info


    def generate_env(self):
        self.width = 20
        self.height = 20

        rail_gen = comm_rail_generator()

        '''
            sparse_rail_generator(
                max_num_cities=2,
                grid_mode=True,
                max_rails_between_cities=1,
                max_rails_in_city=1,
                seed=1),
        '''

        self.env = RailEnv(
            20, 
            20,
            rail_generator=rail_gen,
            malfunction_generator_and_process_data=malfunction_from_params(self.stochastic_data),
            schedule_generator = self.schedule_gen,
            number_of_agents=2,
            obs_builder_object=self.observation_builder,
            remove_agents_at_target=True,
            random_seed=None)

        self.env.global_reward = self.global_reward
        self.env.num_agents = self.num_agents
        self.env.step_penalty = self.initial_step_penalty
        return self.env

    def get_agents_count(self):
        return self.env.num_agents

    def get_env(self):
        return self.env

    def update_env_with_params(self, width, height, num_agents, max_steps, rail_type, rail_gen_params, seed=-1):
        if seed == -1:
            seed=random.randint(0,100000)

        self.num_agents = num_agents
        self.max_steps = max_steps

        if rail_type == 'complex':
            self.rail_gen = complex_rail_generator(
                nr_start_goal=rail_gen_params['nr_start_goal'],
                nr_extra=rail_gen_params['nr_extra'],
                min_dist=rail_gen_params['min_dist'],
                max_dist=rail_gen_params['max_dist'],
                seed=seed
            )

            #self.schedule_gen = complex_schedule_generator()
        elif rail_type == 'sparse':
            self.rail_gen = sparse_rail_generator(
                max_num_cities=rail_gen_params['num_cities'],
                seed=seed,
                grid_mode=rail_gen_params['grid_mode'],
                max_rails_between_cities=rail_gen_params['max_rails_between_cities'],
                max_rails_in_city=rail_gen_params['max_rails_in_city']
            )
            
        else:
            raise ValueError('Please specify either "complex" or "sparse" as rail_type')

        self.generate_env()
            

