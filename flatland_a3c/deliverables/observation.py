"""
On basis of the work of S. Huschauer and the Flatland-Environment-Tree-Observation
"""

import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
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
        
        # Tree
        super().__init__()
        self.max_depth = max_depth
        self.observation_dim = 11
        # Compute the size of the returned observation vector
        size = 0
        pow4 = 1
        for i in range(self.max_depth + 1):
            size += pow4
            pow4 *= 4
        self.tree_observation_space = [size * self.observation_dim]
        self.location_has_agent = {}
        self.location_has_agent_direction = {}
        self.predictor = ShortestPathPredictorForRailEnv()
        self.agents_previous_reset = None
        self.tree_explored_actions = [1, 2, 3, 0]
        self.tree_explorted_actions_char = ['L', 'F', 'R', 'B']
        self.distance_map = None
        self.distance_map_computed = False

        # Own
        #self.reset()
        self.size_ = size_
        self.observation_space = np.zeros((6,size_[0],size_[1]))

    def _set_env(self, env):
        self.env = env
        if self.predictor:
            self.predictor._set_env(self.env)

    def reset(self):
        """
        Called after each environment reset.
        """
        self.map_ = None
        self.agent_positions_ = None
        self.agent_handles_ = None
        self.is_reset = True

        # Tree
        agents = self.env.agents
        nb_agents = len(agents)
        compute_distance_map = True
        if self.agents_previous_reset is not None and nb_agents == len(self.agents_previous_reset):
            compute_distance_map = False
            for i in range(nb_agents):
                if agents[i].target != self.agents_previous_reset[i].target:
                    compute_distance_map = True
        # Don't compute the distance map if it was loaded
        if self.agents_previous_reset is None and self.distance_map is not None:
            self.location_has_target = {tuple(agent.target): 1 for agent in agents}
            compute_distance_map = False

        if compute_distance_map:
            self._compute_distance_map()

        self.agents_previous_reset = agents

    def _compute_distance_map(self):
        agents = self.env.agents
        # For testing only --> To assert if a distance map need to be recomputed.
        self.distance_map_computed = True
        nb_agents = len(agents)
        self.distance_map = np.inf * np.ones(shape=(nb_agents,
                                                    self.env.height,
                                                    self.env.width,
                                                    4))
        self.max_dist = np.zeros(nb_agents)
        self.max_dist = [self._distance_map_walker(agent.target, i) for i, agent in enumerate(agents)]
        # Update local lookup table for all agents' target locations
        self.location_has_target = {tuple(agent.target): 1 for agent in agents}

    def _distance_map_walker(self, position, target_nr):
        """
        Utility function to compute distance maps from each cell in the rail network (and each possible
        orientation within it) to each agent's target cell.
        """
        # Returns max distance to target, from the farthest away node, while filling in distance_map
        self.distance_map[target_nr, position[0], position[1], :] = 0

        # Fill in the (up to) 4 neighboring nodes
        # direction is the direction of movement, meaning that at least a possible orientation of an agent
        # in cell (row,col) allows a movement in direction `direction'
        nodes_queue = deque(self._get_and_update_neighbors(position, target_nr, 0, enforce_target_direction=-1))

        # BFS from target `position' to all the reachable nodes in the grid
        # Stop the search if the target position is re-visited, in any direction
        visited = {(position[0], position[1], 0), (position[0], position[1], 1), (position[0], position[1], 2),
                   (position[0], position[1], 3)}

        max_distance = 0

        while nodes_queue:
            node = nodes_queue.popleft()

            node_id = (node[0], node[1], node[2])

            if node_id not in visited:
                visited.add(node_id)

                # From the list of possible neighbors that have at least a path to the current node, only keep those
                # whose new orientation in the current cell would allow a transition to direction node[2]
                valid_neighbors = self._get_and_update_neighbors((node[0], node[1]), target_nr, node[3], node[2])

                for n in valid_neighbors:
                    nodes_queue.append(n)

                if len(valid_neighbors) > 0:
                    max_distance = max(max_distance, node[3] + 1)

        return max_distance

    def _get_and_update_neighbors(self, position, target_nr, current_distance, enforce_target_direction=-1):
        """
        Utility function used by _distance_map_walker to perform a BFS walk over the rail, filling in the
        minimum distances from each target cell.
        """
        neighbors = []

        possible_directions = [0, 1, 2, 3]
        if enforce_target_direction >= 0:
            # The agent must land into the current cell with orientation `enforce_target_direction'.
            # This is only possible if the agent has arrived from the cell in the opposite direction!
            possible_directions = [(enforce_target_direction + 2) % 4]

        for neigh_direction in possible_directions:
            new_cell = self._new_position(position, neigh_direction)

            if new_cell[0] >= 0 and new_cell[0] < self.env.height and new_cell[1] >= 0 and new_cell[1] < self.env.width:

                desired_movement_from_new_cell = (neigh_direction + 2) % 4

                # Check all possible transitions in new_cell
                for agent_orientation in range(4):
                    # Is a transition along movement `desired_movement_from_new_cell' to the current cell possible?
                    is_valid = self.env.rail.get_transition((new_cell[0], new_cell[1], agent_orientation),
                                                            desired_movement_from_new_cell)

                    if is_valid:
                        """
                        # TODO: check that it works with deadends! -- still bugged!
                        movement = desired_movement_from_new_cell
                        if isNextCellDeadEnd:
                            movement = (desired_movement_from_new_cell+2) % 4
                        """
                        new_distance = min(self.distance_map[target_nr, new_cell[0], new_cell[1], agent_orientation],
                                           current_distance + 1)
                        neighbors.append((new_cell[0], new_cell[1], agent_orientation, new_distance))
                        self.distance_map[target_nr, new_cell[0], new_cell[1], agent_orientation] = new_distance

        return neighbors

    def _new_position(self, position, movement):
        """
        Utility function that converts a compass movement over a 2D grid to new positions (r, c).
        """
        if movement == Grid4TransitionsEnum.NORTH:
            return (position[0] - 1, position[1])
        elif movement == Grid4TransitionsEnum.EAST:
            return (position[0], position[1] + 1)
        elif movement == Grid4TransitionsEnum.SOUTH:
            return (position[0] + 1, position[1])
        elif movement == Grid4TransitionsEnum.WEST:
            return (position[0], position[1] - 1)

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

    def sigmoid_shifted(self,x):
        return 1 / (1 + math.exp(-5-x))

    def get_many(self, handles=[]):
        for agent in self.env.agents:
            rail_grid = np.zeros_like(self.env.rail.grid, dtype=np.uint16)
            path = a_star(self.env.rail.transitions, rail_grid, agent.position,agent.target)
            agent.path_to_target = path

        self.path_priority_map = np.zeros(self.env.rail.grid.shape)

        if handles is None:
            handles = []
        if self.predictor:
            self.max_prediction_depth = 0
            self.predicted_pos = {}
            self.predicted_dir = {}
            self.predictions = self.predictor.get(custom_args={'distance_map': self.distance_map})
            if self.predictions:

                for t in range(len(self.predictions[0])):
                    pos_list = []
                    dir_list = []
                    for a in handles:
                        pos_list.append(self.predictions[a][t][1:3])
                        dir_list.append(self.predictions[a][t][3])
                    self.predicted_pos.update({t: coordinate_to_position(self.env.width, pos_list)})
                    self.predicted_dir.update({t: dir_list})
                self.max_prediction_depth = len(self.predicted_pos)

        observations = {}
        for h in handles:
            observations[h] = self.get(h)

        return self.reshape_obs(super().get_many(handles=handles))

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
        
        if self.is_reset:
            self.is_reset = False
            self.layers = np.zeros((16,self.map_size[0],self.map_size[1]))
            for l in range(16):
                shift = 15 - l
                self.layers[l,:,:] = (self.env.rail.grid >> shift) & 1

        agents = self.env.agents
        agent = agents[handle]
        target = agent.target
        position = agent.position
        speed = agent.speed_data['speed']
        last_action =  agent.speed_data['transition_action_on_cellexit']/4.0
        
        self.pos = np.array(list(agent.position))
        self.offset = np.floor(np.divide(self.size_,2))

        '''
        # Layer with position of agent
        position_map = self.tuples_to_grid([(agent.position[0],agent.position[1], self.convert_dir(agent.direction))])
        layer_position_map = self.to_obs_space(position_map)

        # Layer with speed of agent
        speed_map = self.tuples_to_grid([(agent.position[0],agent.position[1], speed)])
        layer_speed_map = self.to_obs_space(speed_map)

        # Layer with speed of agent
        last_action_map = self.tuples_to_grid([(agent.position[0],agent.position[1], last_action)])
        layer_last_action_map = self.to_obs_space(last_action_map)
        '''

        # Layer with position of agent
        target_map = self.tuples_to_grid([(agent.target[0],agent.target[1])])
        layer_target_map = self.to_obs_space(target_map)

        # Layer with path to target
        path_map = self.tuples_to_grid(agent.path_to_target, True)
        layer_path_to_target = self.to_obs_space(path_map)

        self.env.dev_obs_dict[handle] = agent.path_to_target

        path_priority_map = path_map*self.handle_to_prio(handle)
        self.path_priority_map = np.maximum(path_priority_map, self.path_priority_map)
        layer_path_priority = self.to_obs_space(self.path_priority_map)

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
            layer_path_to_target,
            layer_path_priority,
            layer_agent_positions_map,
            layer_agent_priority,
            layer_agent_target_map,
            layer_grid_map
        ])

        # Grid layout with 16 x obs-size x obs-size
        layer_grid = self.to_obs_space(self.layers, (16,self.size_[0],self.size_[1]))

        # Vector with train-info
        vector = [speed, last_action, self.sigmoid_shifted(len(agent.path_to_target)),self.convert_dir(agent.direction), self.env._elapsed_steps]
        
        # Tree
        """
        Computes the current observation for agent `handle' in env

        The observation vector is composed of 4 sequential parts, corresponding to data from the up to 4 possible
        movements in a RailEnv (up to because only a subset of possible transitions are allowed in RailEnv).
        The possible movements are sorted relative to the current orientation of the agent, rather than NESW as for
        the transitions. The order is:
            [data from 'left'] + [data from 'forward'] + [data from 'right'] + [data from 'back']

        Each branch data is organized as:
            [root node information] +
            [recursive branch data from 'left'] +
            [... from 'forward'] +
            [... from 'right] +
            [... from 'back']

        Each node information is composed of 9 features:

        #1: if own target lies on the explored branch the current distance from the agent in number of cells is stored.

        #2: if another agents target is detected the distance in number of cells from the agents current location
            is stored

        #3: if another agent is detected the distance in number of cells from current agent position is stored.

        #4: possible conflict detected
            tot_dist = Other agent predicts to pass along this cell at the same time as the agent, we store the
             distance in number of cells from current agent position

            0 = No other agent reserve the same cell at similar time

        #5: if an not usable switch (for agent) is detected we store the distance.

        #6: This feature stores the distance in number of cells to the next branching  (current node)

        #7: minimum distance from node to the agent's target given the direction of the agent if this path is chosen

        #8: agent in the same direction
            n = number of agents present same direction
                (possible future use: number of other agents in the same direction in this branch)
            0 = no agent present same direction

        #9: agent in the opposite direction
            n = number of agents present other direction than myself (so conflict)
                (possible future use: number of other agents in other direction in this branch, ie. number of conflicts)
            0 = no agent present other direction than myself

        #10: malfunctioning/blokcing agents
            n = number of time steps the oberved agent remains blocked

        #11: slowest observed speed of an agent in same direction
            1 if no agent is observed

            min_fractional speed otherwise

        Missing/padding nodes are filled in with -inf (truncated).
        Missing values in present node are filled in with +inf (truncated).


        In case of the root node, the values are [0, 0, 0, 0, distance from agent to target, own malfunction, own speed]
        In case the target node is reached, the values are [0, 0, 0, 0, 0].
        """

        # Update local lookup table for all agents' positions
        self.location_has_agent = {tuple(agent.position): 1 for agent in self.env.agents}
        self.location_has_agent_direction = {tuple(agent.position): agent.direction for agent in self.env.agents}
        self.location_has_agent_speed = {tuple(agent.position): agent.speed_data['speed'] for agent in self.env.agents}
        self.location_has_agent_malfunction = {tuple(agent.position): agent.malfunction_data['malfunction'] for agent in
                                               self.env.agents}

        if handle > len(self.env.agents):
            print("ERROR: obs _get - handle ", handle, " len(agents)", len(self.env.agents))
        agent = self.env.agents[handle]  # TODO: handle being treated as index
        possible_transitions = self.env.rail.get_transitions(*agent.position, agent.direction)
        num_transitions = np.count_nonzero(possible_transitions)

        # Root node - current position
        # Here information about the agent itself is stored
        observation = [0, 0, 0, 0, 0, 0, self.distance_map[(handle, *agent.position, agent.direction)], 0, 0,
                       agent.malfunction_data['malfunction'], agent.speed_data['speed']]

        visited = set()

        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right, back], relative to the current orientation
        # If only one transition is possible, the tree is oriented with this transition as the forward branch.
        orientation = agent.direction

        if num_transitions == 1:
            orientation = np.argmax(possible_transitions)

        for branch_direction in [(orientation + i) % 4 for i in range(-1, 3)]:
            if possible_transitions[branch_direction]:
                new_cell = self._new_position(agent.position, branch_direction)
                branch_observation, branch_visited = \
                    self._explore_branch(handle, new_cell, branch_direction, 1, 1)
                observation = observation + branch_observation
                visited = visited.union(branch_visited)
            else:
                # add cells filled with infinity if no transition is possible
                observation = observation + [-np.inf] * self._num_cells_to_fill_in(self.max_depth)
        self.env.dev_obs_dict[handle] = visited


        self.observation_space = [observation_maps, layer_grid, vector,observation]
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
        map_obs = []
        vec_obs = []
        grid_obs = []
        tree_obs = []
        num_agents = len(agent_observations)

        for i in range(num_agents):
            agent_obs = agent_observations[i]
            map_obs.append(agent_obs[0])
            grid_obs.append(agent_obs[1])
            vec_obs.append(agent_obs[2])
            tree_obs.append(agent_obs[3])
        
        map_obs = np.asarray(map_obs)
        map_obs = np.reshape(map_obs,(num_agents, params.map_state_size[0], params.map_state_size[1], params.map_state_size[2]))

        grid_obs = np.asarray(grid_obs)
        grid_obs = np.reshape(grid_obs,(num_agents, params.grid_state_size[0], params.grid_state_size[1], params.grid_state_size[2],1))

        vec_obs = np.asarray(vec_obs)
        tree_obs = np.asarray(tree_obs)
        tree_obs[tree_obs ==  np.inf] = 0.25
        tree_obs[tree_obs ==  -np.inf] = 0.5

        map_obs = map_obs.astype(np.float32)
        grid_obs = grid_obs.astype(np.float32)
        vec_obs = vec_obs.astype(np.float32)
        tree_obs = tree_obs.astype(np.float32)

        return [map_obs, grid_obs, vec_obs, tree_obs]

    def _num_cells_to_fill_in(self, remaining_depth):
            """Computes the length of observation vector: sum_{i=0,depth-1} 2^i * observation_dim."""
            num_observations = 0
            pow4 = 1
            for i in range(remaining_depth):
                num_observations += pow4
                pow4 *= 4
            return num_observations * self.observation_dim

    def _explore_branch(self, handle, position, direction, tot_dist, depth):
        """
        Utility function to compute tree-based observations.
        We walk along the branch and collect the information documented in the get() function.
        If there is a branching point a new node is created and each possible branch is explored.
        """

        # [Recursive branch opened]
        if depth >= self.max_depth + 1:
            return [], []

        # Continue along direction until next switch or
        # until no transitions are possible along the current direction (i.e., dead-ends)
        # We treat dead-ends as nodes, instead of going back, to avoid loops
        exploring = True
        last_is_switch = False
        last_is_dead_end = False
        last_is_terminal = False  # wrong cell OR cycle;  either way, we don't want the agent to land here
        last_is_target = False

        visited = set()
        agent = self.env.agents[handle]
        time_per_cell = np.reciprocal(agent.speed_data["speed"])
        own_target_encountered = np.inf
        other_agent_encountered = np.inf
        other_target_encountered = np.inf
        potential_conflict = np.inf
        unusable_switch = np.inf
        other_agent_same_direction = 0
        other_agent_opposite_direction = 0
        malfunctioning_agent = 0
        min_fractional_speed = 1.
        num_steps = 1
        while exploring:
            # #############################
            # #############################
            # Modify here to compute any useful data required to build the end node's features. This code is called
            # for each cell visited between the previous branching node and the next switch / target / dead-end.
            if position in self.location_has_agent:
                if tot_dist < other_agent_encountered:
                    other_agent_encountered = tot_dist

                # Check if any of the observed agents is malfunctioning, store agent with longest duration left
                if self.location_has_agent_malfunction[position] > malfunctioning_agent:
                    malfunctioning_agent = self.location_has_agent_malfunction[position]

                if self.location_has_agent_direction[position] == direction:
                    # Cummulate the number of agents on branch with same direction
                    other_agent_same_direction += 1

                    # Check fractional speed of agents
                    current_fractional_speed = self.location_has_agent_speed[position]
                    if current_fractional_speed < min_fractional_speed:
                        min_fractional_speed = current_fractional_speed

                if self.location_has_agent_direction[position] != direction:
                    # Cummulate the number of agents on branch with other direction
                    other_agent_opposite_direction += 1

            # Check number of possible transitions for agent and total number of transitions in cell (type)
            cell_transitions = self.env.rail.get_transitions(*position, direction)
            transition_bit = bin(self.env.rail.get_full_transitions(*position))
            total_transitions = transition_bit.count("1")
            crossing_found = False
            if int(transition_bit, 2) == int('1000010000100001', 2):
                crossing_found = True

            # Register possible future conflict
            predicted_time = int(tot_dist * time_per_cell)
            if self.predictor and predicted_time < self.max_prediction_depth:
                int_position = coordinate_to_position(self.env.width, [position])
                if tot_dist < self.max_prediction_depth:

                    pre_step = max(0, predicted_time - 1)
                    post_step = min(self.max_prediction_depth - 1, predicted_time + 1)

                    # Look for conflicting paths at distance tot_dist
                    if int_position in np.delete(self.predicted_pos[predicted_time], handle, 0):
                        conflicting_agent = np.where(self.predicted_pos[predicted_time] == int_position)
                        for ca in conflicting_agent[0]:
                            if direction != self.predicted_dir[predicted_time][ca] and cell_transitions[
                                self._reverse_dir(
                                    self.predicted_dir[predicted_time][ca])] == 1 and tot_dist < potential_conflict:
                                potential_conflict = tot_dist
                            if self.env.dones[ca] and tot_dist < potential_conflict:
                                potential_conflict = tot_dist

                    # Look for conflicting paths at distance num_step-1
                    elif int_position in np.delete(self.predicted_pos[pre_step], handle, 0):
                        conflicting_agent = np.where(self.predicted_pos[pre_step] == int_position)
                        for ca in conflicting_agent[0]:
                            if direction != self.predicted_dir[pre_step][ca] \
                                and cell_transitions[self._reverse_dir(self.predicted_dir[pre_step][ca])] == 1 \
                                and tot_dist < potential_conflict:  # noqa: E125
                                potential_conflict = tot_dist
                            if self.env.dones[ca] and tot_dist < potential_conflict:
                                potential_conflict = tot_dist

                    # Look for conflicting paths at distance num_step+1
                    elif int_position in np.delete(self.predicted_pos[post_step], handle, 0):
                        conflicting_agent = np.where(self.predicted_pos[post_step] == int_position)
                        for ca in conflicting_agent[0]:
                            if direction != self.predicted_dir[post_step][ca] and cell_transitions[self._reverse_dir(
                                self.predicted_dir[post_step][ca])] == 1 \
                                and tot_dist < potential_conflict:  # noqa: E125
                                potential_conflict = tot_dist
                            if self.env.dones[ca] and tot_dist < potential_conflict:
                                potential_conflict = tot_dist

            if position in self.location_has_target and position != agent.target:
                if tot_dist < other_target_encountered:
                    other_target_encountered = tot_dist

            if position == agent.target and tot_dist < own_target_encountered:
                own_target_encountered = tot_dist

            # #############################
            # #############################
            if (position[0], position[1], direction) in visited:
                last_is_terminal = True
                break
            visited.add((position[0], position[1], direction))

            # If the target node is encountered, pick that as node. Also, no further branching is possible.
            if np.array_equal(position, self.env.agents[handle].target):
                last_is_target = True
                break

            # Check if crossing is found --> Not an unusable switch
            if crossing_found:
                # Treat the crossing as a straight rail cell
                total_transitions = 2
            num_transitions = np.count_nonzero(cell_transitions)

            exploring = False

            # Detect Switches that can only be used by other agents.
            if total_transitions > 2 > num_transitions and tot_dist < unusable_switch:
                unusable_switch = tot_dist

            if num_transitions == 1:
                # Check if dead-end, or if we can go forward along direction
                nbits = total_transitions
                if nbits == 1:
                    # Dead-end!
                    last_is_dead_end = True

                if not last_is_dead_end:
                    # Keep walking through the tree along `direction'
                    exploring = True
                    # convert one-hot encoding to 0,1,2,3
                    direction = np.argmax(cell_transitions)
                    position = self._new_position(position, direction)
                    num_steps += 1
                    tot_dist += 1
            elif num_transitions > 0:
                # Switch detected
                last_is_switch = True
                break

            elif num_transitions == 0:
                # Wrong cell type, but let's cover it and treat it as a dead-end, just in case
                print("WRONG CELL TYPE detected in tree-search (0 transitions possible) at cell", position[0],
                      position[1], direction)
                last_is_terminal = True
                break

        # `position' is either a terminal node or a switch

        # #############################
        # #############################
        # Modify here to append new / different features for each visited cell!

        if last_is_target:
            observation = [own_target_encountered,
                           other_target_encountered,
                           other_agent_encountered,
                           potential_conflict,
                           unusable_switch,
                           tot_dist,
                           0,
                           other_agent_same_direction,
                           other_agent_opposite_direction,
                           malfunctioning_agent,
                           min_fractional_speed
                           ]

        elif last_is_terminal:
            observation = [own_target_encountered,
                           other_target_encountered,
                           other_agent_encountered,
                           potential_conflict,
                           unusable_switch,
                           np.inf,
                           self.distance_map[handle, position[0], position[1], direction],
                           other_agent_same_direction,
                           other_agent_opposite_direction,
                           malfunctioning_agent,
                           min_fractional_speed
                           ]

        else:
            observation = [own_target_encountered,
                           other_target_encountered,
                           other_agent_encountered,
                           potential_conflict,
                           unusable_switch,
                           tot_dist,
                           self.distance_map[handle, position[0], position[1], direction],
                           other_agent_same_direction,
                           other_agent_opposite_direction,
                           malfunctioning_agent,
                           min_fractional_speed
                           ]
        # #############################
        # #############################
        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right, back], relative to the current orientation
        # Get the possible transitions
        possible_transitions = self.env.rail.get_transitions(*position, direction)
        for branch_direction in [(direction + 4 + i) % 4 for i in range(-1, 3)]:
            if last_is_dead_end and self.env.rail.get_transition((*position, direction),
                                                                 (branch_direction + 2) % 4):
                # Swap forward and back in case of dead-end, so that an agent can learn that going forward takes
                # it back
                new_cell = self._new_position(position, (branch_direction + 2) % 4)
                branch_observation, branch_visited = self._explore_branch(handle,
                                                                          new_cell,
                                                                          (branch_direction + 2) % 4,
                                                                          tot_dist + 1,
                                                                          depth + 1)
                observation = observation + branch_observation
                if len(branch_visited) != 0:
                    visited = visited.union(branch_visited)
            elif last_is_switch and possible_transitions[branch_direction]:
                new_cell = self._new_position(position, branch_direction)
                branch_observation, branch_visited = self._explore_branch(handle,
                                                                          new_cell,
                                                                          branch_direction,
                                                                          tot_dist + 1,
                                                                          depth + 1)
                observation = observation + branch_observation
                if len(branch_visited) != 0:
                    visited = visited.union(branch_visited)
            else:
                # no exploring possible, add just cells with infinity
                observation = observation + [-np.inf] * self._num_cells_to_fill_in(self.max_depth - depth)

        return observation, visited

    def util_print_obs_subtree(self, tree):
        """
        Utility function to pretty-print tree observations returned by this object.
        """
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.unfold_observation_tree(tree))

    def unfold_observation_tree(self, tree, current_depth=0, actions_for_display=True):
        """
        Utility function to pretty-print tree observations returned by this object.
        """
        if len(tree) < self.observation_dim:
            return

        depth = 0
        tmp = len(tree) / self.observation_dim - 1
        pow4 = 4
        while tmp > 0:
            tmp -= pow4
            depth += 1
            pow4 *= 4

        unfolded = {}
        unfolded[''] = tree[0:self.observation_dim]
        child_size = (len(tree) - self.observation_dim) // 4
        for child in range(4):
            child_tree = tree[(self.observation_dim + child * child_size):
                              (self.observation_dim + (child + 1) * child_size)]
            observation_tree = self.unfold_observation_tree(child_tree, current_depth=current_depth + 1)
            if observation_tree is not None:
                if actions_for_display:
                    label = self.tree_explorted_actions_char[child]
                else:
                    label = self.tree_explored_actions[child]
                unfolded[label] = observation_tree
        return unfolded

    def _reverse_dir(self, direction):
        return int((direction + 2) % 4)
