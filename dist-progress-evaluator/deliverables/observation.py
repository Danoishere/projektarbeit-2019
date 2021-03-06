
import numpy as np
import deliverables.input_params as params
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

"""
Collection of environment-specific ObservationBuilder.
"""
import collections
from typing import Optional, List, Dict, Tuple

import numpy as np

from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.agent_utils import RailAgentStatus, EnvAgent
from flatland.utils.ordered_set import OrderedSet


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
                                          'communication '
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

        # Update local lookup table for all agents' positions
        # ignore other agents not in the grid (only status active and done)
        # self.location_has_agent = {tuple(agent.position): 1 for agent in self.env.agents if
        #                         agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.DONE]}

        self.location_has_agent = {}
        self.location_has_agent_direction = {}
        self.location_has_agent_speed = {}
        self.location_has_agent_malfunction = {}
        self.location_has_agent_ready_to_depart = {}
        self.location_has_agent_communication = {}

        for _agent in self.env.agents:
            if _agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.DONE] and \
                    _agent.position:

                try:
                    _agent.communication
                except:
                    _agent.communication = None

                self.location_has_agent[tuple(_agent.position)] = 1
                self.location_has_agent_direction[tuple(_agent.position)] = _agent.direction
                self.location_has_agent_speed[tuple(_agent.position)] = _agent.speed_data['speed']
                self.location_has_agent_malfunction[tuple(_agent.position)] = _agent.malfunction_data['malfunction']
                self.location_has_agent_communication[tuple(_agent.position)] = _agent.communication

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
                                                       communication=np.zeros(5),
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
        found_closest_communication = False
        communication = None

        while exploring:
            # #############################
            # #############################
            # Modify here to compute any useful data required to build the end node's features. This code is called
            # for each cell visited between the previous branching node and the next switch / target / dead-end.
            if position in self.location_has_agent:
                if self.location_has_agent_communication[position] is not None and not found_closest_communication:
                    found_closest_communication = True,
                    communication = self.location_has_agent_communication[position]

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
                                      communication=communication,
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
        super().__init__(params.tree_depth, ShortestPathPredictorForRailEnv())
        self.last_obs = {}
        self.actor_rec_state = {}
        self.critic_rec_state = {}
        self.comm_rec_state = {}

    def reset(self):
        self.last_obs = {}
        self.actor_rec_state = {}
        self.critic_rec_state = {}
        self.comm_rec_state = {}
        return super().reset()

    def get_many(self, handles=None):
        obs = super().get_many(handles=handles)
        all_augmented_obs = {}

        for handle in obs:
            last_agent_obs = None
            if handle in self.last_obs:
                last_agent_obs = self.last_obs[handle]

            next_agent_obs = obs[handle]
            next_agent_tree_obs, vec_obs, comm_obs, rec_actor, rec_critic, rec_comm = self.reshape_agent_obs(handle, next_agent_obs, None)
            # print(next_agent_tree_obs)
            #augmented_tree_obs = self.augment_agent_tree_obs_with_frames(last_agent_obs, next_agent_tree_obs)
            self.last_obs[handle] = next_agent_tree_obs# augmented_tree_obs

            # print(augmented_tree_obs)
            all_augmented_obs[handle] = (next_agent_tree_obs, vec_obs, comm_obs, rec_actor, rec_critic, rec_comm)

        return all_augmented_obs

    def reshape_agent_obs(self, handle, agent_obs, info):
        if agent_obs is None:
            # New tree-obs is just the size of one frame
            tree_obs = np.zeros(params.frame_size)
            vec_obs = np.zeros(params.vec_state_size)
            comm_obs = np.zeros(params.comm_size)

            if handle in self.actor_rec_state and handle in self.critic_rec_state:
                agent_actor_rec_state = self.actor_rec_state[handle]
                agent_critic_rec_state = self.critic_rec_state[handle]
                agent_comm_rec_state = self.comm_rec_state[handle]
            else:
                agent_actor_rec_state = np.zeros((2,params.recurrent_size)).astype(np.float32)
                agent_critic_rec_state = np.zeros((2,params.recurrent_size)).astype(np.float32)
                agent_comm_rec_state = np.zeros((2,params.recurrent_size)).astype(np.float32)

            return tree_obs.astype(np.float32), vec_obs.astype(np.float32), comm_obs.astype(np.float32),  agent_actor_rec_state, agent_critic_rec_state, agent_comm_rec_state
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
                if agent_obs == -np.inf:
                    raise ValueError('Well, looks like it\'s not always s straight')

            # Fastest way from root
            # fastest way: tree-depth + 1 (root)
            fastest_way = get_shortest_way_from('.',agent_obs, params.tree_depth +1)
            sorted_children = get_ordered_children(agent_obs)

            # second fastest way: tree-depth
            alt_way_1 = [None]* params.tree_depth
            # Try to take second best solution at next intersection
            if len(sorted_children) > 1:
                alt_node_1 = sorted_children[1]
                alt_way_1 = get_shortest_way_from(alt_node_1[0], alt_node_1[1], params.tree_depth)

            alt_way_2 = [None]*(params.tree_depth-1)
            # Try to take second best solution at second next intersection
            if fastest_way[1] != None:
                sorted_children = get_ordered_children(fastest_way[1][1])
                if len(sorted_children) > 1:
                    alt_node_2 = sorted_children[1]
                    alt_way_2 = get_shortest_way_from(alt_node_2[0],alt_node_2[1], params.tree_depth-1)

            obs_layers = [fastest_way, alt_way_1, alt_way_2]
            tree_obs = []
            comm_obs = []
            for layer in obs_layers:
                for node in layer:
                    node_obs = node_to_obs(node)
                    tree_obs.append(node_obs)
                    agent_comm_obs = node_to_comm(node)
                    comm_obs.append(agent_comm_obs)

            tree_obs = np.concatenate(tree_obs)
            comm_obs = np.concatenate(comm_obs)

            agent = self.env.agents[handle]
            # print(agent.position)

            # Current info about the train itself
            vec_obs = np.zeros(params.vec_state_size)

            if agent.moving:
                vec_obs[0] = 1.0

            vec_obs[1] = agent.malfunction_data['malfunction']
            vec_obs[2] = agent.speed_data['speed']
            vec_obs[3] = normalize_field(agent.status.value, 5)
            vec_obs[4] = normalize_field(agent.direction, 4.0)

            if agent.position is not None and agent.target is not None:
                dist_x = np.abs(agent.target[0] - agent.position[0])
                vec_obs[5] = normalize_field(dist_x)
                dist_y = np.abs(agent.target[1] - agent.position[1])
                vec_obs[6] = normalize_field(dist_y)

            vec_obs[7] = normalize_field(root_node.dist_min_to_target)

            if handle in self.actor_rec_state and handle in self.critic_rec_state:
                agent_actor_rec_state = self.actor_rec_state[handle]
                agent_critic_rec_state = self.critic_rec_state[handle]
                agent_comm_rec_state = self.comm_rec_state[handle]
            else:
                agent_actor_rec_state = np.zeros((2,params.recurrent_size)).astype(np.float32)
                agent_critic_rec_state = np.zeros((2,params.recurrent_size)).astype(np.float32)
                agent_comm_rec_state = np.zeros((2,params.recurrent_size)).astype(np.float32)

            return tree_obs.astype(np.float32), vec_obs.astype(np.float32), comm_obs.astype(np.float32),  agent_actor_rec_state, agent_critic_rec_state, agent_comm_rec_state


    def augment_agent_tree_obs_with_frames(self, last_obs, next_obs):
        single_obs_len = params.frame_size
        full_obs_len = params.tree_state_size

        if last_obs is None:
            last_obs = np.zeros(full_obs_len)

        last_multi_frame_obs = last_obs[:-single_obs_len]
        multi_frame_obs = np.zeros(full_obs_len)

        # Start = new obs
        multi_frame_obs[:single_obs_len] = next_obs

        # Fill remaining n-1 slots with last obs
        multi_frame_obs[single_obs_len:single_obs_len+len(last_multi_frame_obs)] = last_multi_frame_obs
        return multi_frame_obs.astype(np.float32)


def buffer_to_obs_lists(episode_buffer):
    tree_obs = np.asarray([row[0][0] for row in episode_buffer])
    vec_obs = np.asarray([row[0][1] for row in episode_buffer])
    comm_obs = np.asarray([row[0][2] for row in episode_buffer])
    a_rec_obs = np.asarray([row[0][3] for row in episode_buffer])
    c_rec_obs = np.asarray([row[0][4] for row in episode_buffer])
    comm_rec_obs = np.asarray([row[0][5] for row in episode_buffer])

    return tree_obs, vec_obs, comm_obs, a_rec_obs, c_rec_obs, comm_rec_obs


# Observation-pattern
# ####################
#
# |-|-|-|-|-|-|  <- 1.) Direct route to target
#   |-|-|-|-|-|  <- 2.) Route to target with different decision at next intersection
#     |-|-|-|-|  <- 3.) Route to target with different decision at second next intersection
#
# Put all in one vector


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
    #if node is None:
    #    return []

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

def node_to_comm(node_tuple):
    if node_tuple is None:
        return [0]*params.number_of_comm

    dir = node_tuple[0]
    node = node_tuple[1]

    if node.communication is None:
        return [0]*params.number_of_comm

    return node.communication


def node_to_obs(node_tuple):
    if node_tuple is None:
        return [0]*params.num_features

    dir = node_tuple[0]
    node = node_tuple[1]

    dir_dict = {
        '.' : 0.1,
        'F': 0.4,
        'L': 0.6,
        'R': 0.7,
    }

    dist_left = 0
    dist_right = 0
    dist_forward = 0

    # Is there a way that goes left and goes to the target at the end of
    if 'L' in node.childs and node.childs['L'] != -np.inf:
        dist_left = normalize_field(node.childs['L'].dist_min_to_target)
    if 'R' in node.childs and node.childs['R'] != -np.inf:
        dist_right = normalize_field(node.childs['R'].dist_min_to_target)
    if 'F' in node.childs and node.childs['F'] != -np.inf:
        dist_forward = normalize_field(node.childs['F'].dist_min_to_target)

    dir_num = dir_dict[dir]
    obs = [
        1 if dir == '.' else 0,
        1 if dir == 'F' else 0,
        1 if dir == 'L' else 0,
        1 if dir == 'R' else 0,
        dir_num,
        normalize_field(node.dist_min_to_target),
        normalize_field(node.dist_other_agent_encountered),
        one_hot(node.dist_other_agent_encountered),
        one_hot(node.dist_other_target_encountered),
        #normalize_field(node.dist_other_target_encountered),
        #normalize_field(node.dist_own_target_encountered),
        one_hot(node.dist_own_target_encountered),
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
        dist_left,
        dist_right,
        dist_forward,
        0,
        0,
        0,
        0,
        0
    ]

    if node.communication is not None:
        obs[-5:] = node.communication

    return obs
