"""
On basis of the work of S. Huschauer and the Flatland-Environment-Tree-Observation
"""

import numpy as np
import pandas as pd

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.observations import TreeObsForRailEnv
from flatland.core.grid.grid4_astar import a_star
from flatland.core.transition_map import GridTransitionMap
from numpy.core.umath import divide
import deliverables.input_params as params
import math

import pprint
from collections import deque
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

class CombinedObservation(ObservationBuilder):

    def __init__(self, size_, max_depth):
        self.tree = TreeObsForRailEnv(max_depth, ShortestPathPredictorForRailEnv())

        # Own
        #self.reset()
        self.size_ = size_
        self.observation_space = np.zeros((6,size_[0],size_[1]))

    def _set_env(self, env):
        self.env = env
        self.tree._set_env(env)

    def reset(self):
        """
        Called after each environment reset.
        """
        self.map_ = None
        self.agent_positions_ = None
        self.agent_handles_ = None
        self.is_reset = True
        self.tree.reset()

    def handle_to_prio(self, handle):
        return (handle + 1)/15


    def convert_grid(self, grid_):
        """
        transition_list = [int('0000000000000000', 2),  # empty cell - Case 0
                       int('1000000000100000', 2),  # Case 1 - straight
                       int('1001001000100000', 2),  # Case 2 - simple switch
                       int('1000010000100001', 2),  # Case 3 - diamond drossing
                       int('1001011000100001', 2),  # Case 4 - single slip
                       int('1100110000110011', 2),  # Case 5 - double slip
                       int('0101001000000010', 2),  # Case 6 - symmetrical
                       int('0010000000000000', 2),  # Case 7 - dead end
                       int('0100000000000010', 2),  # Case 1b (8)  - simple turn right
                       int('0001001000000000', 2),  # Case 1c (9)  - simple turn left
                       int('1100000000100010', 2)]  # Case 2b (10) - simple switch mirrored
        """
        map_ = np.zeros_like(grid_).astype(np.float)
        map_[grid_==int('1000000000100000', 2)] = 0.3
        map_[grid_==int('1001001000100000', 2)] = 0.35
        map_[grid_==int('1000010000100001', 2)] = 0.4
        map_[grid_==int('1001011000100001', 2)] = 0.45
        map_[grid_==int('1100110000110011', 2)] = 0.5
        map_[grid_==int('0101001000000010', 2)] = 0.55
        map_[grid_==int('0010000000000000', 2)] = 0.6
        map_[grid_==int('0100000000000010', 2)] = 0.65
        map_[grid_==int('0001001000000000', 2)] = 0.7
        map_[grid_==int('1100000000100010', 2)] = 0.75
  
        return map_

    def sigmoid_shifted(self,x):
        return 1 / (1 + math.exp(-5-x))

    def sigmoid(self,x, derivative=False):
        return x*(1-x) if derivative else 1/(1+np.exp(-x))

    def get_many(self, handles=[]):
        '''
        for agent in self.env.agents:
            rail_grid = np.zeros_like(self.env.rail.grid, dtype=np.uint16)
            path = a_star(self.env.rail.transitions, rail_grid, agent.position,agent.target)
            agent.path_to_target = path

        self.path_priority_map = np.zeros(self.env.rail.grid.shape)
        '''

        tree_obs = self.tree.get_many(handles)

        df = pd.DataFrame()

        obs0 = []
        obs1 = []
        obs2 = []

        for h in handles:
            obs = self.get(h)

            t_obs = np.asarray(tree_obs[h])
            t_obs[t_obs ==  np.inf] = -1
            t_obs[t_obs ==  -np.inf] = -2
            t_obs += 2
            t_obs /= 40.0

            t_obs = np.concatenate([t_obs, obs[2]])

            obs0.append(np.reshape(obs[0], params.map_state_size).astype(np.float32))
            obs1.append(np.reshape(obs[1], params.grid_state_size).astype(np.float32))
            obs2.append(t_obs.astype(np.float32))

        return obs0, obs1, obs2

    def get(self, handle=0):
        """

        Called whenever an observation has to be computed for the `env' environment, possibly
        for each agent independently (agent id `handle').
 
        Parameters
        -------
        handle : int (optional)
            Handle of the agent for which to compute the observation vector.
 
        Returns
        -------
        function
        Transition map as local window of size x,y , agent-positions if in window and target.
        """


        self.offset_initialized = False

        grid_map = self.convert_grid(self.env.rail.grid)
        self.map_size = grid_map.shape
        
        '''
        if self.is_reset:
            self.is_reset = False
            self.layers = np.zeros((16,self.map_size[0],self.map_size[1]))
            for l in range(16):
                shift = 15 - l
                self.layers[l,:,:] = (self.env.rail.grid >> shift) & 1
        '''

        agents = self.env.agents
        agent = agents[handle]
        target = agent.target
        position = agent.position
        speed = agent.speed_data['speed']
        last_action =  agent.speed_data['transition_action_on_cellexit']/4.0
        
        self.pos = np.array(list(agent.position))
        self.offset = np.floor(np.divide(self.size_,2))

        
        # Layer with position of agent
        position_map = self.tuples_to_grid([(agent.position[0],agent.position[1], self.convert_dir(agent.direction))])
        layer_position_map = self.to_obs_space(position_map)
        # Layer with speed of agent
        speed_map = self.tuples_to_grid([(agent.position[0],agent.position[1], speed)])
        layer_speed_map = self.to_obs_space(speed_map)
        # Layer with speed of agent
        last_action_map = self.tuples_to_grid([(agent.position[0],agent.position[1], last_action)])
        layer_last_action_map = self.to_obs_space(last_action_map)
        

        # Layer with position of agent
        target_map = self.tuples_to_grid([(agent.target[0],agent.target[1])])
        layer_target_map = self.to_obs_space(target_map)

        # Layer with path to target
        '''
        path_map = self.tuples_to_grid(agent.path_to_target, True)
        layer_path_to_target = self.to_obs_space(path_map)
        self.env.dev_obs_dict[handle] = agent.path_to_target
        
        path_priority_map = path_map*self.handle_to_prio(handle)
        self.path_priority_map = np.maximum(path_priority_map, self.path_priority_map)
        layer_path_priority = self.to_obs_space(self.path_priority_map)
        '''

        # Targets for other agents & their positions
        agent_targets = []
        agent_positions = []
        agent_priority = []
        agent_speed = []
        agent_action_on_cellexit = []
        for agent in agents:
            if agent.handle != handle:
                agent_targets.append(agent.target)
                agent_positions.append((
                    agent.position[0],
                    agent.position[1],
                    self.convert_dir(agent.direction)))

            agent_speed.append((
                agent.position[0],
                agent.position[1],
                agent.speed_data['speed']))

            agent_action_on_cellexit.append((
                agent.position[0],
                agent.position[1],
                agent.speed_data['transition_action_on_cellexit']/4.0))
                
            agent_priority.append((
                agent.position[0],
                agent.position[1],
                self.handle_to_prio(handle)))

        if len(agent_targets) > 0:
            target_map = self.tuples_to_grid(agent_targets)
            layer_agent_target_map = self.to_obs_space(target_map)
            agent_positions_map = self.tuples_to_grid(agent_positions)
            layer_agent_positions_map = self.to_obs_space(agent_positions_map)
        else:
            layer_agent_target_map = np.zeros(self.size_)
            layer_agent_positions_map = np.zeros(self.size_)

        priority_map = self.tuples_to_grid(agent_priority)
        layer_agent_priority = self.to_obs_space(priority_map)

        speed_map = self.tuples_to_grid(agent_speed)
        layer_agent_speed = self.to_obs_space(speed_map)

        action_map = self.tuples_to_grid(agent_action_on_cellexit)
        layer_agent_action = self.to_obs_space(action_map)
        layer_grid_map = self.to_obs_space(grid_map)

        observation_maps = np.array([
            layer_agent_speed,
            layer_agent_action,
            layer_target_map, 
            layer_agent_positions_map,
            layer_agent_priority,
            layer_agent_target_map,
            layer_grid_map
        ])

        grid_flat = self.convert_grid(self.env.rail.grid)

        # Grid layout with 16 x obs-size x obs-size
        layer_grid = self.to_obs_space(grid_flat)

        # Vector with train-info
        vector = [speed, last_action, 0,self.convert_dir(agent.direction), self.env._elapsed_steps]

        self.observation_space = [observation_maps, layer_grid, vector]
        return self.observation_space


    def tuples_to_grid(self,tuples_list,increase_with_dist=False):
        # Tuples in format (y,x,val)
        if len(tuples_list[0]) == 3:
            obs_grid = np.zeros(self.map_size)
            for tp in tuples_list:
                obs_grid[tp[0],tp[1]] = tp[2]
            return obs_grid
        # Tuples in format (y,x)
        else:
            obs_grid = np.zeros(self.map_size)
            cnt = 0
            dist = len(tuples_list)
            for tp in tuples_list:
                if increase_with_dist:
                    obs_grid[tp] = (dist-cnt+1)/(dist+1)
                    cnt += 1
                else:
                    obs_grid[tp] = 1.0
                
            return obs_grid


    def to_obs_space(self, orig_map,obs_size=-1):

        use_default_size = False
        if obs_size == -1:
            use_default_size = True
            obs_size = self.size_

        if not self.offset_initialized:
            orig_size = orig_map.shape

            grid_offset_min = self.pos - self.offset
            grid_offset_max = self.pos + self.offset + 1

            self.min_map = np.maximum(grid_offset_min,[0,0]).astype(np.int16)
            self.max_map = np.minimum(grid_offset_max, orig_size).astype(np.int16)

            grid_offset_min = self.offset - self.pos
            grid_offset_max = self.offset + orig_size - self.pos

            self.min_obs = np.maximum(grid_offset_min,[0,0]).astype(np.int16)
            self.max_obs = np.minimum(grid_offset_max, obs_size).astype(np.int16)
            self.offset_initialized = True

    
        # Default obs_size
        if use_default_size:
            copied_area = orig_map[
                self.min_map[0]:self.max_map[0], 
                self.min_map[1]:self.max_map[1]]
            obs_grid = np.zeros(self.size_)
        else:
            copied_area = orig_map[:,
                self.min_map[0]:self.max_map[0], 
                self.min_map[1]:self.max_map[1]]

            obs_grid = np.zeros(obs_size)
            obs_grid[:,
                self.min_obs[0]:self.max_obs[0], 
                self.min_obs[1]:self.max_obs[1]] = copied_area

        return obs_grid

    def convert_dir(self, direction):
        return float(direction + 1)/10.0

    def reshape_obs(self, agent_observations):
        tree_obs = []
        num_agents = len(agent_observations)

        for i in range(num_agents):
            agent_obs = agent_observations[i]
            tree_obs.append(agent_obs[0])

        tree_obs = np.asarray(tree_obs)
        tree_obs[tree_obs ==  np.inf] = -15.0
        tree_obs[tree_obs ==  -np.inf] = 20.0
        tree_obs = tree_obs.astype(np.float32)
        
        tree_obs = tree_obs*0.5
        tree_obs = tree_obs - 5
        tree_obs = self.sigmoid(tree_obs)

        return [tree_obs]