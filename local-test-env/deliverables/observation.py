import numpy as np
import deliverables.input_params as params
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

"""
Collection of environment-specific ObservationBuilder.
"""
import collections
from typing import Optional, List, Dict, Tuple

import numpy as np
from time import time

from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.agent_utils import RailAgentStatus, EnvAgent
from flatland.utils.ordered_set import OrderedSet
from flatland.envs.rail_env import RailEnvActions


class CustomTreeObsForRailEnv(ObservationBuilder):
    """
    TreeObsForRailEnv object.

    This object returns observation vectors for agents in the RailEnv environment.
    The information is local to each agent and exploits the graph structure of the rail
    network to simplify the representation of the state of the environment for each agent.

    For details about the features in the tree observation see the get() function.
    """
    Node = collections.namedtuple('Node', 'dist_own_target_encountered '
                                          'dist_other_target_encountered '
                                          'dist_other_agent_encountered '
                                          'dist_potential_conflict '
                                          'dist_unusable_switch '
                                          'dist_to_next_branch '
                                          'dist_min_to_target '
                                          'num_agents_same_direction '
                                          'num_agents_opposite_direction '
                                          'num_agents_malfunctioning '
                                          'speed_min_fractional '
                                          'num_agents_ready_to_depart '
                                          'other_agents '
                                          'childs')

    tree_explored_actions_char = ['L', 'F', 'R', 'B']

    def __init__(self, max_depth: int, predictor: PredictionBuilder = None):
        super().__init__()
        self.max_depth = max_depth
        self.observation_dim = 11
        self.location_has_agent = {}
        self.location_has_agent_direction = {}
        self.predictor = predictor
        self.location_has_target = None

    def reset(self):
        self.location_has_target = {tuple(agent.target): 1 for agent in self.env.agents}

    def get_many(self, handles: Optional[List[int]] = None) -> Dict[int, Node]:
        """
        Called whenever an observation has to be computed for the `env` environment, for each agent with handle
        in the `handles` list.
        """

        if handles is None:
            handles = []
        if self.predictor:
            self.max_prediction_depth = 0
            self.predicted_pos = {}
            self.predicted_dir = {}
            self.predictions = self.predictor.get()
            if self.predictions:
                for t in range(self.predictor.max_depth + 1):
                    pos_list = []
                    dir_list = []
                    for a in handles:
                        if self.predictions[a] is None:
                            continue
                        pos_list.append(self.predictions[a][t][1:3])
                        dir_list.append(self.predictions[a][t][3])
                    self.predicted_pos.update({t: coordinate_to_position(self.env.width, pos_list)})
                    self.predicted_dir.update({t: dir_list})
                self.max_prediction_depth = len(self.predicted_pos)
        
        observations = super().get_many(handles)
        return observations

    def get(self, handle: int = 0) -> Node:
        """
        Computes the current observation for agent `handle` in env

        The observation vector is composed of 4 sequential parts, corresponding to data from the up to 4 possible
        movements in a RailEnv (up to because only a subset of possible transitions are allowed in RailEnv).
        The possible movements are sorted relative to the current orientation of the agent, rather than NESW as for
        the transitions. The order is::

            [data from 'left'] + [data from 'forward'] + [data from 'right'] + [data from 'back']

        Each branch data is organized as::

            [root node information] +
            [recursive branch data from 'left'] +
            [... from 'forward'] +
            [... from 'right] +
            [... from 'back']

        Each node information is composed of 9 features:

        #1:
            if own target lies on the explored branch the current distance from the agent in number of cells is stored.

        #2:
            if another agents target is detected the distance in number of cells from the agents current location\
            is stored

        #3:
            if another agent is detected the distance in number of cells from current agent position is stored.

        #4:
            possible conflict detected
            tot_dist = Other agent predicts to pass along this cell at the same time as the agent, we store the \
             distance in number of cells from current agent position

            0 = No other agent reserve the same cell at similar time

        #5:
            if an not usable switch (for agent) is detected we store the distance.

        #6:
            This feature stores the distance in number of cells to the next branching  (current node)

        #7:
            minimum distance from node to the agent's target given the direction of the agent if this path is chosen

        #8:
            agent in the same direction
            n = number of agents present same direction \
                (possible future use: number of other agents in the same direction in this branch)
            0 = no agent present same direction

        #9:
            agent in the opposite direction
            n = number of agents present other direction than myself (so conflict) \
                (possible future use: number of other agents in other direction in this branch, ie. number of conflicts)
            0 = no agent present other direction than myself

        #10:
            malfunctioning/blokcing agents
            n = number of time steps the oberved agent remains blocked

        #11:
            slowest observed speed of an agent in same direction
            1 if no agent is observed

            min_fractional speed otherwise
        #12:
            number of agents ready to depart but no yet active

        Missing/padding nodes are filled in with -inf (truncated).
        Missing values in present node are filled in with +inf (truncated).


        In case of the root node, the values are [0, 0, 0, 0, distance from agent to target, own malfunction, own speed]
        In case the target node is reached, the values are [0, 0, 0, 0, 0].
        """

        self.location_has_agent = {}
        self.location_has_agent_direction = {}
        self.location_has_agent_speed = {}
        self.location_has_agent_malfunction = {}
        self.location_has_agent_ready_to_depart = {}
        self.location_has_agent_obj = {}

        for _agent in self.env.agents:
            if _agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.DONE] and \
                    _agent.position:

                self.location_has_agent_obj[tuple(_agent.position)] = _agent
                self.location_has_agent[tuple(_agent.position)] = 1
                self.location_has_agent_direction[tuple(_agent.position)] = _agent.direction
                self.location_has_agent_speed[tuple(_agent.position)] = _agent.speed_data['speed']
                self.location_has_agent_malfunction[tuple(_agent.position)] = _agent.malfunction_data['malfunction']

            if _agent.status in [RailAgentStatus.READY_TO_DEPART] and \
                    _agent.initial_position:
                self.location_has_agent_ready_to_depart[tuple(_agent.initial_position)] = \
                    self.location_has_agent_ready_to_depart.get(tuple(_agent.initial_position), 0) + 1

        if handle > len(self.env.agents):
            print("ERROR: obs _get - handle ", handle, " len(agents)", len(self.env.agents))
        agent = self.env.agents[handle]  # TODO: handle being treated as index

        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
        else:
            return None

        possible_transitions = self.env.rail.get_transitions(*agent_virtual_position, agent.direction)
        num_transitions = np.count_nonzero(possible_transitions)

        # Here information about the agent itself is stored
        distance_map = self.env.distance_map.get()

        root_node_observation = CustomTreeObsForRailEnv.Node(dist_own_target_encountered=0, dist_other_target_encountered=0,
                                                       dist_other_agent_encountered=0, dist_potential_conflict=0,
                                                       dist_unusable_switch=0, dist_to_next_branch=0,
                                                       dist_min_to_target=distance_map[
                                                           (handle, *agent_virtual_position,
                                                            agent.direction)],
                                                       num_agents_same_direction=0, num_agents_opposite_direction=0,
                                                       num_agents_malfunctioning=agent.malfunction_data['malfunction'],
                                                       speed_min_fractional=agent.speed_data['speed'],
                                                       num_agents_ready_to_depart=0,
                                                       other_agents=[],
                                                       childs={})

        visited = OrderedSet()

        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right, back], relative to the current orientation
        # If only one transition is possible, the tree is oriented with this transition as the forward branch.
        orientation = agent.direction

        if num_transitions == 1:
            orientation = np.argmax(possible_transitions)

        for i, branch_direction in enumerate([(orientation + i) % 4 for i in range(-1, 3)]):

            if possible_transitions[branch_direction]:
                new_cell = get_new_position(agent_virtual_position, branch_direction)

                branch_observation, branch_visited = \
                    self._explore_branch(handle, new_cell, branch_direction, 1, 1)
                root_node_observation.childs[self.tree_explored_actions_char[i]] = branch_observation

                visited |= branch_visited
            else:
                # add cells filled with infinity if no transition is possible
                root_node_observation.childs[self.tree_explored_actions_char[i]] = -np.inf
        self.env.dev_obs_dict[handle] = visited

        return root_node_observation

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

        visited = OrderedSet()
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
        other_agent_ready_to_depart_encountered = 0
        other_agents_opposite_dir = []

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

                other_agent_ready_to_depart_encountered += self.location_has_agent_ready_to_depart.get(position, 0)

                if self.location_has_agent_direction[position] == direction:
                    # Cummulate the number of agents on branch with same direction
                    other_agent_same_direction += self.location_has_agent_direction.get((position, direction), 0)

                    # Check fractional speed of agents
                    current_fractional_speed = self.location_has_agent_speed[position]
                    if current_fractional_speed < min_fractional_speed:
                        min_fractional_speed = current_fractional_speed

                    # Other direction agents
                    # TODO: Test that this behavior is as expected
                    other_agent_opposite_direction += \
                        self.location_has_agent[position] - self.location_has_agent_direction.get((position, direction),
                                                                                                  0)

                else:
                    other_agents_opposite_dir.append(self.location_has_agent_obj[position])
                    # If no agent in the same direction was found all agents in that position are other direction
                    other_agent_opposite_direction += self.location_has_agent[position]

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
                            if self.env.agents[ca].status == RailAgentStatus.DONE and tot_dist < potential_conflict:
                                potential_conflict = tot_dist

                    # Look for conflicting paths at distance num_step-1
                    elif int_position in np.delete(self.predicted_pos[pre_step], handle, 0):
                        conflicting_agent = np.where(self.predicted_pos[pre_step] == int_position)
                        for ca in conflicting_agent[0]:
                            if direction != self.predicted_dir[pre_step][ca] \
                                and cell_transitions[self._reverse_dir(self.predicted_dir[pre_step][ca])] == 1 \
                                and tot_dist < potential_conflict:  # noqa: E125
                                potential_conflict = tot_dist
                            if self.env.agents[ca].status == RailAgentStatus.DONE and tot_dist < potential_conflict:
                                potential_conflict = tot_dist

                    # Look for conflicting paths at distance num_step+1
                    elif int_position in np.delete(self.predicted_pos[post_step], handle, 0):
                        conflicting_agent = np.where(self.predicted_pos[post_step] == int_position)
                        for ca in conflicting_agent[0]:
                            if direction != self.predicted_dir[post_step][ca] and cell_transitions[self._reverse_dir(
                                self.predicted_dir[post_step][ca])] == 1 \
                                and tot_dist < potential_conflict:  # noqa: E125
                                potential_conflict = tot_dist
                            if self.env.agents[ca].status == RailAgentStatus.DONE and tot_dist < potential_conflict:
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
                    # Keep walking through the tree along `direction`
                    exploring = True
                    # convert one-hot encoding to 0,1,2,3
                    direction = np.argmax(cell_transitions)
                    position = get_new_position(position, direction)
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

        # `position` is either a terminal node or a switch

        # #############################
        # #############################
        # Modify here to append new / different features for each visited cell!

        if last_is_target:
            dist_to_next_branch = tot_dist
            dist_min_to_target = 0
        elif last_is_terminal:
            dist_to_next_branch = np.inf
            dist_min_to_target = self.env.distance_map.get()[handle, position[0], position[1], direction]
        else:
            dist_to_next_branch = tot_dist
            dist_min_to_target = self.env.distance_map.get()[handle, position[0], position[1], direction]

        node = CustomTreeObsForRailEnv.Node(dist_own_target_encountered=own_target_encountered,
                                      dist_other_target_encountered=other_target_encountered,
                                      dist_other_agent_encountered=other_agent_encountered,
                                      dist_potential_conflict=potential_conflict,
                                      dist_unusable_switch=unusable_switch,
                                      dist_to_next_branch=dist_to_next_branch,
                                      dist_min_to_target=dist_min_to_target,
                                      num_agents_same_direction=other_agent_same_direction,
                                      num_agents_opposite_direction=other_agent_opposite_direction,
                                      num_agents_malfunctioning=malfunctioning_agent,
                                      speed_min_fractional=min_fractional_speed,
                                      num_agents_ready_to_depart=other_agent_ready_to_depart_encountered,
                                      other_agents=other_agents_opposite_dir,
                                      childs={})

        # #############################
        # #############################
        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right, back], relative to the current orientation
        # Get the possible transitions
        possible_transitions = self.env.rail.get_transitions(*position, direction)
        for i, branch_direction in enumerate([(direction + 4 + i) % 4 for i in range(-1, 3)]):
            if last_is_dead_end and self.env.rail.get_transition((*position, direction),
                                                                 (branch_direction + 2) % 4):
                # Swap forward and back in case of dead-end, so that an agent can learn that going forward takes
                # it back
                new_cell = get_new_position(position, (branch_direction + 2) % 4)
                branch_observation, branch_visited = self._explore_branch(handle,
                                                                          new_cell,
                                                                          (branch_direction + 2) % 4,
                                                                          tot_dist + 1,
                                                                          depth + 1)
                node.childs[self.tree_explored_actions_char[i]] = branch_observation
                if len(branch_visited) != 0:
                    visited |= branch_visited
            elif last_is_switch and possible_transitions[branch_direction]:
                new_cell = get_new_position(position, branch_direction)
                branch_observation, branch_visited = self._explore_branch(handle,
                                                                          new_cell,
                                                                          branch_direction,
                                                                          tot_dist + 1,
                                                                          depth + 1)
                node.childs[self.tree_explored_actions_char[i]] = branch_observation
                if len(branch_visited) != 0:
                    visited |= branch_visited
            else:
                # no exploring possible, add just cells with infinity
                node.childs[self.tree_explored_actions_char[i]] = -np.inf

        if depth == self.max_depth:
            node.childs.clear()
        return node, visited

    def util_print_obs_subtree(self, tree: Node):
        """
        Utility function to print tree observations returned by this object.
        """
        self.print_node_features(tree, "root", "")
        for direction in self.tree_explored_actions_char:
            self.print_subtree(tree.childs[direction], direction, "\t")

    @staticmethod
    def print_node_features(node: Node, label, indent):
        print(indent, "Direction ", label, ": ", node.dist_own_target_encountered, ", ",
              node.dist_other_target_encountered, ", ", node.dist_other_agent_encountered, ", ",
              node.dist_potential_conflict, ", ", node.dist_unusable_switch, ", ", node.dist_to_next_branch, ", ",
              node.dist_min_to_target, ", ", node.num_agents_same_direction, ", ", node.num_agents_opposite_direction,
              ", ", node.num_agents_malfunctioning, ", ", node.speed_min_fractional, ", ",
              node.num_agents_ready_to_depart)

    def print_subtree(self, node, label, indent):
        if node == -np.inf or not node:
            print(indent, "Direction ", label, ": -np.inf")
            return

        self.print_node_features(node, label, indent)

        if not node.childs:
            return

        for direction in self.tree_explored_actions_char:
            self.print_subtree(node.childs[direction], direction, indent + "\t")

    def set_env(self, env: Environment):
        super().set_env(env)
        if self.predictor:
            self.predictor.set_env(self.env)

    def _reverse_dir(self, direction):
        return int((direction + 2) % 4)


class RailObsBuilder(CustomTreeObsForRailEnv):
    def __init__(self):
        super().__init__(params.tree_depth)
        self.actor_rec_state = {}
        self.critic_rec_state = {}
        self.prep_step = 0
        self.start_groups = None

    def reset(self):
        self.actor_rec_state = {}
        self.critic_rec_state = {}
        self.prep_step = 0
        self.start_groups = None
        return super().reset()

    def get_many(self, handles=None):
        start = time()

        # Set default values
        agents = self.env.agents
        for agent in agents:
            try:
                agent.activate
            except:
                agent.activate = False

        num_agents = len(agents)
        num_active_agents = 0
        num_ready_agents = 0
        for agent in agents:
            if agent.status == RailAgentStatus.ACTIVE:
                num_active_agents += 1
            if agent.status == RailAgentStatus.READY_TO_DEPART:
                num_ready_agents += 1

        progress = self.env._elapsed_steps/self.env._max_episode_steps
        departure_rate = np.min([progress*1.3,  1.0])
        num_target_active_agents = np.max([1, int(num_agents * departure_rate)])
        # print('Target: ', num_target_active_agents, progress)

        next_to_activate = None
        largest_dist_sum = 100
        speed_of_largest_dist = 0

        if self.start_groups is None:
            self.start_groups = {}
            for agent in agents:
                if agent.initial_position not in self.start_groups:
                    self.start_groups[agent.initial_position] = [agent]
                else:
                    self.start_groups[agent.initial_position].append(agent)


            for key in self.start_groups:
                self.start_groups[key] = sorted(self.start_groups[key], key=lambda x: x.speed_data['speed'], reverse=True)

            '''
            clean_start_groups = {}
            for key in self.start_groups:
                for key2 in self.start_groups:
                    if key != key2:
                        diff_y = abs(key[0] - key2[0])
                        diff_x = abs(key[1] - key2[1])

                        if (diff_x + diff_y) <= 3:
                            keys = tuple(sorted([key, key2]))
                            clean_start_groups[keys] = self.start_groups[key] + self.start_groups[key2]
            '''


            def speed_sum(group):
                sum = 0
                for agent in group:
                    sum += agent.speed_data['speed']
                return sum
                
            # speed_sum(g)

            self.start_groups = sorted(self.start_groups.values(), key=lambda g: len(g), reverse=True)


        active = 0
        inactive = 0
        for agent in agents:
            if agent.activate:
                active += 1
            else:
                inactive += 1

        #print('Created')
        if active < num_target_active_agents:
            num_activate = num_target_active_agents - num_active_agents
            agent_to_activate = self.start_groups[0].pop(0)
            agent_to_activate.activate = True
        
            if len(self.start_groups[0]) == 0:
                self.start_groups.pop(0)
                
            

            print(active, '/', inactive)
            '''
                agent.dist_sum = 0
                if agent.status == RailAgentStatus.READY_TO_DEPART:
                    posy = agent.initial_position[0]
                    posx = agent.initial_position[1]
                    agent.dist_sum = 0
                    for other_agent in agents:
                        if other_agent.status == RailAgentStatus.ACTIVE:
                            dist = np.abs(other_agent.initial_position[0] - posy)+np.abs(other_agent.initial_position[1] - posx)
                            agent.dist_sum += dist * agent.speed_data['speed']
                        
                if (agent.dist_sum <= largest_dist_sum): # or (agent.dist_sum == largest_dist_sum and speed_of_largest_dist < agent.speed_data['speed']):
                    largest_dist_sum = agent.dist_sum
                    speed_of_largest_dist = agent.speed_data['speed']
                    next_to_activate = agent.handle

            '''
                
        #if next_to_activate is not None:
        #    agents[next_to_activate].activate = True

        actions = {}
        for agent in agents:
            try:
                agent.wait
                agent.wait = np.max([agent.wait - 1,0]) 
            except:
                agent.wait = 0

            agent.next_pos, agent.next_dir = self.next_pos(agent.position, agent.direction)

            agent.is_on_unusable_switch = self.is_agent_on_unusable_switch(agent.position, agent.direction)
            agent.is_on_usable_switch = self.is_agent_on_usable_switch(agent.position, agent.direction)
            agent.is_next_unusable_switch = self.is_agent_on_unusable_switch(agent.next_pos, agent.next_dir)
            agent.is_next_usable_switch = self.is_agent_on_usable_switch(agent.next_pos, agent.next_dir)

            '''
            print('----------------------------------')
            print('curr unus:', agent.is_on_unusable_switch)
            print('curr us:', agent.is_on_usable_switch)
            print('nxt unus:', agent.is_next_unusable_switch)
            print('nxt us:', agent.is_next_usable_switch)
            print('----------------------------------')
            '''

            if agent.status == RailAgentStatus.READY_TO_DEPART:
                if agent.activate:
                    actions[agent.handle] = RailEnvActions.MOVE_FORWARD
                else:
                    actions[agent.handle] = RailEnvActions.DO_NOTHING

            elif agent.wait > 0 and agent.moving > 0:
                actions[agent.handle] = RailEnvActions.STOP_MOVING
            elif agent.wait > 0 and not agent.moving > 0:
                actions[agent.handle] = RailEnvActions.DO_NOTHING
            elif agent.malfunction_data['malfunction'] > 0:
                actions[agent.handle] = RailEnvActions.DO_NOTHING
            elif agent.is_next_unusable_switch:
                pass 
            elif agent.is_next_usable_switch:
                actions[agent.handle] = RailEnvActions.MOVE_FORWARD 
            elif not agent.is_on_usable_switch:
                actions[agent.handle] = RailEnvActions.MOVE_FORWARD

        obs_dict = {}
        for agent in agents:
            if (self.env.action_required(agent) and agent.handle not in actions) or \
                (agent.activate and agent.status == RailAgentStatus.READY_TO_DEPART):

                agent_obs = self.get(agent.handle)
                agent_obs, rec_actor, rec_critic = self.reshape_agent_obs(agent.handle, agent_obs, None)
                obs_dict[agent.handle] = (agent_obs, rec_actor, rec_critic)

                if agent.activate and agent.status == RailAgentStatus.READY_TO_DEPART:
                    root = agent.tree_obs[0][0][1]
                    if len(root.other_agents) > 0:
                        actions[agent.handle] = RailEnvActions.DO_NOTHING

                if agent.is_on_usable_switch:
                    root = agent.tree_obs[0][0][1]

                    left = root.childs['L']
                    forward = root.childs['F']
                    right = root.childs['R']

                    if left == -np.inf:
                        left = forward
                    if right == -np.inf:
                        right = forward

                    if len(left.other_agents) > 0 and len(right.other_agents) == 0:
                        if right == -np.inf:
                            actions[agent.handle] = RailEnvActions.MOVE_FORWARD
                        else:
                            actions[agent.handle] = RailEnvActions.MOVE_RIGHT

                    if len(left.other_agents) == 0 and len(right.other_agents)  > 0:
                        if left == -np.inf:
                            actions[agent.handle] = RailEnvActions.MOVE_FORWARD
                        else:
                            actions[agent.handle] = RailEnvActions.MOVE_LEFT

                    elif len(left.other_agents) > 0 and len(right.other_agents)  > 0:
                        if left.other_agents[0].handle == right.other_agents[0].handle:
                            actions[agent.handle] = RailEnvActions.MOVE_LEFT
                        else:
                            actions[agent.handle] = RailEnvActions.STOP_MOVING

                if agent.is_next_unusable_switch:
                    root = agent.tree_obs[0][0][1]
                    if len(root.other_agents) > 0:
                        actions[agent.handle] = RailEnvActions.STOP_MOVING
                    else:
                        left = root.childs['L']
                        forward = root.childs['F']
                        right = root.childs['R']

                        if left == -np.inf:
                            left = forward
                        if right == -np.inf:
                            right = forward

                        if left != -np.inf and right != -np.inf:
                            if len(left.other_agents) > 0 and len(right.other_agents)  > 0:
                                if left.other_agents[0].handle != right.other_agents[0].handle:
                                    actions[agent.handle] = RailEnvActions.STOP_MOVING

        self.env.next_actions = actions
        return obs_dict

    
    def get_all_obs(self):
        obs_dict = {}
        for agent in self.env.agents:
            if agent.status == RailAgentStatus.READY_TO_DEPART or agent.status == RailAgentStatus.ACTIVE:
                agent_obs = self.get(agent.handle)
                agent_obs, rec_actor, rec_critic = self.reshape_agent_obs(agent.handle, agent_obs, None)
                obs_dict[agent.handle] = (agent_obs, rec_actor, rec_critic)

        return obs_dict
    
    
    def binary_tree(self, root_node):
        depth = params.tree_depth - 1

        node_info = {
            'is_root' : True,
            'closer' : True,
            'dist' : root_node.dist_to_next_branch + root_node.dist_min_to_target,
            'turn' : '.'
        }

        depth_list = [[(node_info, root_node)]]
        for d in range(depth):
            depth_list.append([])
            for n in depth_list[d]:
                left,left_turn, right, right_turn = self.get_turns(n)
                if left is None and right is None:
                    depth_list[d+1].append((None, None))
                    depth_list[d+1].append((None, None))
                else:
                    is_left_closer, is_right_closer, dist_left, dist_right = self.get_closer_turn(left, right)

                    left_info = {
                        'is_root' : False,
                        'closer' : is_left_closer,
                        'dist' : dist_left,
                        'turn' : left_turn
                    }

                    right_info = {
                        'is_root' : False,
                        'closer' : is_right_closer,
                        'dist' : dist_right,
                        'turn' : right_turn
                    }

                    depth_list[d+1].append((left_info, left))
                    depth_list[d+1].append((right_info, right))

        return depth_list

    def get_turns(self, node_tuple):
        if node_tuple is None:
            return None, None, None, None

        node_info = node_tuple[0]
        node = node_tuple[1]

        if node is None or node.childs is None or len(node.childs) == 0:
            return None, None, None, None

        left = node.childs['L']
        left_dir = 'L'
        forward = node.childs['F']
        right = node.childs['R']
        right_dir = 'R'

        left_node = None
        right_node = None

        if left != -np.inf:
            left_node = left

        if right != -np.inf:
            right_node = right

        if forward != -np.inf:
            if right_node == None:
                right_node = forward
                right_dir = 'F'
            elif left_node == None:
                left_node = forward
                left_dir = 'F'

        return left_node, left_dir, right_node, right_dir



    def get_closer_turn(self, left, right):
        is_left_closer = False
        is_right_closer = False

        dist_left = np.inf
        dist_right = np.inf
    
        if left is not None and right is not None:
            dist_left = left.dist_to_next_branch + left.dist_min_to_target
            dist_right = right.dist_to_next_branch + right.dist_min_to_target
            if dist_left < dist_right:
                is_left_closer = True
            elif dist_left > dist_right:
                is_right_closer = True
            else:
                is_left_closer = True
                is_right_closer = True

        elif left is None:
            is_right_closer = True
        elif right is None:
            is_left_closer = True

        return is_left_closer, is_right_closer, dist_left, dist_right
        

    def reshape_agent_obs(self, handle, agent_obs, info):
        if agent_obs is None:
            # New tree-obs is just the size of one frame
            tree_obs = np.zeros(params.tree_state_size)
            vec_obs = np.zeros(params.vec_state_size)

            if handle in self.actor_rec_state and handle in self.critic_rec_state:
                agent_actor_rec_state = self.actor_rec_state[handle]
                agent_critic_rec_state = self.critic_rec_state[handle]
            else:
                agent_actor_rec_state = np.zeros((2,params.recurrent_size)).astype(np.float32)
                agent_critic_rec_state = np.zeros((2,params.recurrent_size)).astype(np.float32)

            return np.concatenate([tree_obs, vec_obs]).astype(np.float32),  agent_actor_rec_state, agent_critic_rec_state
        else:

            root_node = agent_obs
            # If we are not at a switch, there is an additional node that only contains the
            # dist. to target. Skip that one and take its first child as root
            num_root_children = 0
            for turn in agent_obs.childs:
                if agent_obs.childs[turn] != -np.inf:
                    num_root_children += 1

            if num_root_children == 1:
                agent_obs = agent_obs.childs['F']

            tree = self.binary_tree(agent_obs)
            agent = self.env.agents[handle]
            agent.tree_obs = tree
            tree_obs = []
            for layer in tree:
                for node in layer:
                    node_obs = node_to_obs(node, agent)
                    tree_obs.append(node_obs)

            tree_obs = np.concatenate(tree_obs)

            

            # Current info about the train itself
            vec_obs = np.zeros(params.vec_state_size)

            vec_obs[0] = agent.moving
            vec_obs[1] = agent.malfunction_data['malfunction']
            vec_obs[2] = agent.speed_data['speed']
            vec_obs[3] = normalize_field(agent.status.value, 5)
            vec_obs[4] = normalize_field(agent.direction, 4.0)

            if agent.position is not None and agent.target is not None:
                dist_x = np.abs(agent.target[0] - agent.position[0])
                vec_obs[5] = normalize_field(dist_x)
                dist_y = np.abs(agent.target[1] - agent.position[1])
                vec_obs[6] = normalize_field(dist_y)

            try:
                agent.is_on_unusable_switch
            except:
                agent.is_on_unusable_switch = False

            try:
                agent.is_on_usable_switch
            except:
                agent.is_on_usable_switch = False

            try:
                agent.is_next_unusable_switch
            except:
                agent.is_next_unusable_switch = False

            try:
                agent.is_next_usable_switch
            except:
                agent.is_next_usable_switch = False

            vec_obs[7] = normalize_field(root_node.dist_min_to_target)
            vec_obs[8] = 1 if agent.is_on_unusable_switch else 0
            vec_obs[9] = 1 if agent.is_on_usable_switch else 0
            vec_obs[10] = 1 if agent.is_next_unusable_switch else 0
            vec_obs[11] = 1 if agent.is_next_usable_switch else 0
            vec_obs[12] = 1 if agent.status == RailAgentStatus.READY_TO_DEPART else 0
            vec_obs[13] = 1 if agent.status == RailAgentStatus.ACTIVE else 0

            if handle in self.actor_rec_state and handle in self.critic_rec_state:
                agent_actor_rec_state = self.actor_rec_state[handle]
                agent_critic_rec_state = self.critic_rec_state[handle]
            else:
                agent_actor_rec_state = np.zeros((2,params.recurrent_size)).astype(np.float32)
                agent_critic_rec_state = np.zeros((2,params.recurrent_size)).astype(np.float32)

            return np.concatenate([tree_obs, vec_obs]).astype(np.float32),  agent_actor_rec_state, agent_critic_rec_state

    def is_agent_on_usable_switch(self, position, dir):
        ''' a tile is a switch with more than one possible transitions for the
            given direction. '''

        if position is None:
            return False

        transition = self.env.rail.get_transitions(*position, dir)

        if np.sum(transition) == 1:
            return False
        else:
            #print(transition)
            return True

    def is_agent_on_unusable_switch(self, position, dir):
        ''' a tile is a switch with more than one possible transitions for the
            given direction. '''

        if position is None:
            return False

        possible_transitions = np.sum(self.env.rail.get_transitions(*position, dir))
        #print(env.rail.get_transitions(*position, dir))
        for d in range(4):
            dir_transitions = np.sum(self.env.rail.get_transitions(*position, d))
            if dir_transitions > possible_transitions >= 1:
                #print(env.rail.get_transitions(*position, d))
                return True

        return False



    def next_pos(self, position, direction):
        if position is None:
            return None, None

        # print('Curr. pos:', position, 'dir', direction)

        transition = self.env.rail.get_transitions(*position, direction)
        if np.sum(transition) > 1:
            return None, None

        posy = position[0] - transition[0]  + transition[2]
        posx = position[1] + transition[1] - transition[3]

        new_dir = np.argmax(transition)
        # print('Next pos:', [posy, posx], 'dir', new_dir)
        return [posy, posx], new_dir


def buffer_to_obs_lists(episode_buffer):
    vec_obs = np.asarray([row[0][0] for row in episode_buffer])
    a_rec_obs = np.asarray([row[0][1] for row in episode_buffer])
    c_rec_obs = np.asarray([row[0][2] for row in episode_buffer])

    return vec_obs, a_rec_obs, c_rec_obs



def get_shortest_way_from(entry_dir, start_node, length):
    selected_nodes = []
    selected_nodes.append((entry_dir,start_node)) # Root

    shortest_way_idx = 'INIT'
    shortest_way = 1000
    found_target = False

    current_node = start_node
    while shortest_way_idx != 'NA' and not found_target and current_node is not None:
        shortest_way_idx = 'NA'
        for k in current_node.childs:
            child = current_node.childs[k]
            if child != -np.inf:
                if child.dist_own_target_encountered != 0 and child.dist_own_target_encountered < 1000:
                    found_target = True
                    shortest_way_idx = k
                elif child.dist_min_to_target < shortest_way and not found_target:
                    shortest_way = child.dist_min_to_target
                    shortest_way_idx = k

        if shortest_way_idx != 'NA':
            next_node = current_node.childs[shortest_way_idx]
            selected_nodes.append((shortest_way_idx, next_node))
            current_node = next_node

    for j in range(len(selected_nodes), length):
        selected_nodes.append(None)

    return selected_nodes


def get_ordered_children(node):
    children = []
    for k in node.childs:
        child = node.childs[k]
        if child != -np.inf:
            children.append((k,child))

    children = sorted(children, key=lambda t: np.min([t[1].dist_min_to_target, t[1].dist_own_target_encountered]))
    return children


def normalize_field(field, norm_val=100):
    if field == np.inf or field == -np.inf:
        return 0
    else:
        return (field+1.0)/norm_val


def one_hot(field):
    if field == np.inf or field == -np.inf or field == 0.0:
        return 0
    else:
        return 1.0


def node_to_obs(node_tuple, agent):
    if node_tuple[1] is None:
        return [0]*params.num_features

    node_info = node_tuple[0]
    node = node_tuple[1]

    dist_left = 0
    dist_right = 0
    dist_forward = 0

    # Is there a way that goes left and goes to the target at the end of
    if 'L' in node.childs and node.childs['L'] != -np.inf:
        dist_left = node.childs['L'].dist_min_to_target
    if 'R' in node.childs and node.childs['R'] != -np.inf:
        dist_right = node.childs['R'].dist_min_to_target
    if 'F' in node.childs and node.childs['F'] != -np.inf:
        dist_forward = node.childs['F'].dist_min_to_target

    target_encountered = node.dist_own_target_encountered != np.inf and node.dist_own_target_encountered != 0

    node_dir = node_info['turn']
    obs = [
        1.0,
        1.0 if node_dir == '.' else 0,
        1.0 if node_dir == 'F' else 0,
        1.0 if node_dir == 'L' else 0,
        1.0 if node_dir == 'R' else 0,
        1.0 if node_info['closer'] else 0,
        np.tanh(node_info['dist']*0.02),
        normalize_field(node.dist_min_to_target),
        normalize_field(node.dist_other_agent_encountered),
        one_hot(node.dist_other_agent_encountered),
        one_hot(node.dist_other_target_encountered),
        1.0 if target_encountered else 0,
        normalize_field(node.dist_potential_conflict),
        one_hot(node.dist_potential_conflict),
        normalize_field(node.dist_to_next_branch),
        normalize_field(node.dist_unusable_switch),
        one_hot(node.dist_unusable_switch),
        normalize_field(node.num_agents_malfunctioning,10),
        normalize_field(node.num_agents_opposite_direction, 10),
        one_hot(node.num_agents_opposite_direction),
        normalize_field(node.num_agents_ready_to_depart, 20),
        normalize_field(node.num_agents_same_direction, 20),
        node.speed_min_fractional,
        normalize_field(dist_left),
        one_hot(dist_left),
        normalize_field(dist_right),
        one_hot(dist_right),
        normalize_field(dist_forward),
        one_hot(dist_forward),
        0,
        0,
        0,
        0,
        0
    ]

    if len(node.other_agents) > 0:
        clostest_agent = node.other_agents[0]
        agent_action_onehot = np.zeros(4)

        try:
            clostest_agent.last_action
        except:
            clostest_agent.last_action = 0

        agent_action_onehot[np.arange(0,4) == clostest_agent.last_action] = 1
        obs[-5] = clostest_agent.moving
        obs[-4:] = agent_action_onehot

    return obs
