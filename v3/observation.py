"""
Author: S.Huschauer
Date: 31.08.2019
Email: huschste@students.zhaw.ch
Associated with ZHAW datalab
This code is not exhaustive and "is as it is"!
"""

import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_astar import a_star
from flatland.core.transition_map import GridTransitionMap
from numpy.core.umath import divide

class RawObservation(ObservationBuilder):
    """
    ObservationBuilder for raw observations.
    """
    def __init__(self, size_):
        self.reset()
        self.size_ = size_
        self.observation_space = np.zeros((6,size_[0],size_[1]))
        self.offset_initialized = False

    def _set_env(self, env):
        self.env = env
 
    def reset(self):
        """
        Called after each environment reset.
        """
        self.map_ = None
        self.agent_positions_ = None
        self.agent_handles_ = None
        
    def handle_to_prio(self, handle):
        return (handle + 1)/15
 
    def get_target_vec(self, target, position_,dir_):

        target = np.array(target)
        position = np.array(position_)
        vec_ = target - position #check this
        dist_ = np.sqrt(vec_.dot(vec_))
        if dist_ == 0:
            return np.array([0.0,0.0,0.0,0.0,0.0,0.0])
        dist_ = np.array([dist_])
        dir_ = np.array([dir_])
        pos_ = np.array([position]).flatten()
        return np.concatenate([vec_/dist_, dist_,pos_,dir_]).flatten()

    def slice_map(self, position):

        shape_ = self.env.rail.grid.shape
        x0 = position[0] - self.size_[0]/2
        x1 = position[1] - self.size_[1]/2
        b0 = self.size_[0]
        b1 = self.size_[1]

        y0 = 0
        y1 = 0

        if x0 < 0:
            y0 -= x0
            b0 += x0
            x0 = 0
        if x0 + b0 >= shape_[0]:
            b0  =  shape_[0] - x0

        if x1 < 0:
            y1 -= x1
            b1 += x1
            x1 = 0
        if x1 + b1 >= shape_[1]:
            b1  =  shape_[1] - x1
        

        return int(x0),int(x1),int(y0),int(y1),int(b0),int(b1)

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

    def get_many(self, handles=[]):
        for agent in self.env.agents:
            rail_grid = np.zeros_like(self.env.rail.grid, dtype=np.uint16)
            path = a_star(self.env.rail.transitions, rail_grid, agent.position,agent.target)
            agent.path_to_target = path

        self.path_priority_map = np.zeros(self.env.rail.grid.shape)
        return super().get_many(handles=handles)

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
       
        grid_map = self.convert_grid(self.env.rail.grid)
        self.map_size = grid_map.shape
        
        agents = self.env.agents
        agent = agents[handle]
        target = agent.target
        position = agent.position
        
        self.pos = np.array(list(agent.position))
        self.offset = np.floor(np.divide(self.size_,2))

        # Layer with position of agent
        position_map = self.tuples_to_grid([(agent.position[0],agent.position[1], self.convert_dir(agent.direction))])
        layer_position_map = self.to_obs_space(position_map)

        # Layer with position of agent
        target_map = self.tuples_to_grid([(agent.target[0],agent.target[1])])
        layer_target_map = self.to_obs_space(target_map)

        # Layer with path to target
        rail_grid = np.zeros_like(grid_map, dtype=np.uint16)
        path_map = self.tuples_to_grid(agent.path_to_target)
        layer_path_to_target = self.to_obs_space(path_map)

        path_priority_map = path_map*self.handle_to_prio(handle)
        self.path_priority_map = np.maximum(path_priority_map, self.path_priority_map)
        layer_path_priority = self.to_obs_space(self.path_priority_map)

        # Targets for other agents & their positions
        agent_targets = []
        agent_positions = []
        agent_priority = []
        for agent in agents:
            if agent.handle != handle:
                agent_targets.append(agent.target)
                agent_positions.append((
                    agent.position[0],
                    agent.position[1],
                    self.convert_dir(agent.direction)))
                
            agent_priority.append((
                agent.position[0],
                agent.position[1],
                self.handle_to_prio(handle)))

        if len(agent_targets) > 0:
            target_map = self.tuples_to_grid(agent_targets)
            layer_agent_target_map = self.to_obs_space(target_map)
            agent_positions_map = self.tuples_to_grid(agent_positions)
            layer_agent_positions_map = self.to_obs_space(agent_positions_map)

        priority_map = self.tuples_to_grid(agent_priority)
        layer_agent_priority = self.to_obs_space(priority_map)
        layer_grid_map = self.to_obs_space(grid_map)

        self.observation_space = np.array([
            layer_position_map,
            layer_target_map, 
            layer_path_to_target,
            layer_path_priority,
            layer_agent_positions_map,
            layer_agent_priority,
            layer_agent_target_map,
            layer_grid_map
        ])

        return self.observation_space


    def tuples_to_grid(self,tuples_list):
        # Tuples in format (y,x,val)
        if len(tuples_list[0]) == 3:
            obs_grid = np.zeros(self.map_size)
            for tp in tuples_list:
                obs_grid[tp[0],tp[1]] = tp[2]
            return obs_grid
        # Tuples in format (y,x)
        else:
            obs_grid = np.zeros(self.map_size)
            for tp in tuples_list:
                obs_grid[tp] = 1.0
            return obs_grid


    def to_obs_space(self, orig_map):
        obs_grid = np.zeros(self.size_)
        orig_size = orig_map.shape

        grid_offset = self.pos - self.offset
        if not self.offset_initialized:
            self.min_map_y = np.int(np.max([grid_offset[0],0]))
            self.max_map_y = np.int(np.min([orig_size[0] + grid_offset[0] + 1, orig_size[0]]))

            self.min_map_x =np.int(np.max([grid_offset[1],0]))
            self.max_map_x = np.int(np.min([orig_size[1] + grid_offset[1] + 1, orig_size[1]]))

            self.min_obs_y = np.int(np.max([grid_offset[0],0]))
            self.max_obs_y = np.int(np.min([orig_size[0] + grid_offset[0] + 1, orig_size[0]]))

            self.min_obs_x =np.int(np.max([grid_offset[1],0]))
            self.max_obs_x = np.int(np.min([orig_size[1] + grid_offset[1] + 1, orig_size[1]]))
            self.offset_initialized = True

        copied_area = orig_map[self.min_map_y:self.max_map_y, self.min_map_x:self.max_map_x]
        obs_grid[self.min_obs_y:self.max_obs_y, self.min_obs_x:self.max_obs_x] = copied_area
        return obs_grid

    def path_to_obs(size_, offset, pos, path, path_to_target_):
        for point in path:
            p = np.array(list(point)) - pos + offset
            if p[0] >= 0 and p[0] < size_[0] and p[1] >= 0 and p[1] < size_[1]:
                p = p.astype(np.int)
                path_to_target_[p[0],p[1]] = 1

        path_to_target_ = path_to_target_.astype(np.float32)
        return path_to_target_

    def convert_dir(self, direction):
        return float(direction + 1)/10.0
