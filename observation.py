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
from numba import njit, jitclass


class RawObservation(ObservationBuilder):
    """
    ObservationBuilder for raw observations.
    """
    def __init__(self, size_):
        self.reset()
        self.size_ = size_
        self.observation_space = np.zeros((6,size_[0],size_[1]))
    def _set_env(self, env):
        self.env = env
 
    def reset(self):
        """
        Called after each environment reset.
        """
        self.map_ = None
        self.agent_positions_ = None
        self.agent_handles_ = None
 
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
       
        
        map_ = self.convert_grid(self.env.rail.grid)
        
        agents = self.env.agents
        agent = agents[handle]
        target = agent.target
        position = agent.position
        
        x0,x1,y0,y1,b0,b1 = self.slice_map(position)

        smap =  np.zeros((self.size_[0],self.size_[1]),dtype = map_.dtype)
        smap[y0:y0+b0,y1:y1+b1] = map_[x0:x0+b0,x1:x1+b1]

        test_map = np.zeros_like(smap)

        agent_positions_ = np.zeros_like(smap)
        agent_targets_ = np.zeros_like(smap)
        path_to_target_ = np.zeros_like(smap, dtype=np.uint16)
        rail_grid = np.zeros_like(map_, dtype=np.uint16)
        

        path = a_star(self.env.rail.transitions, rail_grid, agent.position,agent.target)
        pos = np.array(list(agent.position))
        offset = np.floor(np.divide(self.size_,2))
        
        path_to_target_ = path_to_obs(self.size_,offset, pos, path, path_to_target_)

        size_half_0 = int(self.size_[0]/2)
        size_half_1 = int(self.size_[1]/2)
        for handle_, agent in enumerate(agents):
            if handle!= handle_:
                dir_ = agent.direction
                tar_ = agent.target
                direction_ = float(dir_ + 1)/10.0
                position_ = agent.position
            
                if position_[0]>= x0 and position_[0]< x0 + b0 \
                and position_[1]>= x1 and position_[1]< x1 + b1:
                    agent_positions_[position_[0]-x0][position_[1]-x1] = direction_

                if tar_[0]>= x0 and tar_[0]< x0 + b0 \
                and tar_[1]>= x1 and tar_[1]< x1 + b1:
                    agent_targets_[tar_[0]-x0][tar_[1]-x1] = 0.7
                    
            else:
                
                if target[0]>= x0 and target[0]< x0 + b0 \
                and target[1]>= x1 and target[1]< x1 + b1:
                    agent_targets_[target[0]-x0][target[1]-x1] = 0.3
                    
 
        my_position_ = np.zeros_like(smap)
        
        direction = float(agents[handle].direction + 1)/10.0
        my_position_[int(self.size_[0]/2)][int(self.size_[1]/2)] = direction
        agent_positions_[int(self.size_[0]/2)][int(self.size_[1]/2)] = direction
        my_target_ = np.zeros_like(smap)
        if target[0]>= x0 and target[0]< x0 + b0 \
                and target[1]>= x1 and target[1]< x1 + b1:
                    my_target_[target[0]-x0][target[1]-x1] = 0.5                                                                    
        
      
        self.observation_space = np.stack(( smap,agent_positions_,agent_targets_,my_position_,my_target_,path_to_target_))
        self.observation_space = np.swapaxes(self.observation_space,0,2)
        self.observation_space = [self.observation_space, self.get_target_vec(target,position,direction)]
        return self.observation_space

#@njit
def path_to_obs(size_, offset, pos, path, path_to_target_):
    for point in path:
        p = np.array(list(point)) - pos + offset
        if p[0] >= 0 and p[0] < size_[0] and p[1] >= 0 and p[1] < size_[1]:
            p = p.astype(np.int)
            path_to_target_[p[0],p[1]] = 1

    path_to_target_ = path_to_target_.astype(np.float32)
    return path_to_target_
