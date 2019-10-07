import numpy as np
import math
from typing import Dict, List, Optional, NamedTuple, Tuple, Set, Type,Callable


import numpy as np
from collections import deque

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.transition_map import GridTransitionMap

from flatland.core.env_prediction_builder import PredictionBuilder
from deliverables.utils.agent_status import RailAgentStatus
from flatland.envs.rail_env import RailEnvActions
from deliverables.utils.ordered_set import OrderedSet
from flatland.envs.agent_utils import EnvAgent

RailEnvGridPos = NamedTuple('RailEnvGridPos', [('r', int), ('c', int)])
RailEnvNextAction = NamedTuple('RailEnvNextAction', [('action', RailEnvActions), ('next_position', RailEnvGridPos),
                                                     ('next_direction', Grid4TransitionsEnum)])

WalkingElement = \
    NamedTuple('WalkingElement',
               [('position', Tuple[int, int]), ('direction', int), ('next_action_element', RailEnvNextAction)])

Vector2D: Type = Tuple[float, float]
IntVector2D: Type = Tuple[int, int]

IntVector2DArray: Type = List[IntVector2D]
IntVector2DArrayArray: Type = List[List[IntVector2D]]

Vector2DArray: Type = List[Vector2D]
Vector2DArrayArray: Type = List[List[Vector2D]]

IntVector2DDistance: Type = Callable[[IntVector2D, IntVector2D], float]



def is_dead_end(self, rcPos: IntVector2DArray):
        """
        Check if the cell is a dead-end.

        Parameters
        ----------
        rcPos: Tuple[int,int]
            tuple(row, column) with grid coordinate
        Returns
        -------
        boolean
            True if and only if the cell is a dead-end.
        """
        nbits = 0
        tmp = self.get_full_transitions(rcPos[0], rcPos[1])
        while tmp > 0:
            nbits += (tmp & 1)
            tmp = tmp >> 1
        return nbits == 1


class DistanceMap:
    def __init__(self, agents: List[EnvAgent], env_height: int, env_width: int, rail:Optional[GridTransitionMap]):
        self.env_height = env_height
        self.env_width = env_width
        self.distance_map = None
        self.agents_previous_computation = None
        self.reset_was_called = False
        self.agents: List[EnvAgent] = agents
        self.rail: Optional[GridTransitionMap] = rail
        GridTransitionMap.is_dead_end = is_dead_end

    
    def set(self, distance_map: np.ndarray):
        """
        Set the distance map
        """
        self.distance_map = distance_map

    def get(self) -> np.ndarray:
        """
        Get the distance map
        """
        if self.reset_was_called:
            self.reset_was_called = False

            compute_distance_map = True
            # Don't compute the distance map if it was loaded
            if self.agents_previous_computation is None and self.distance_map is not None:
                compute_distance_map = False

            if compute_distance_map:
                self._compute(self.agents, self.rail)

        elif self.distance_map is None:
            self._compute(self.agents, self.rail)

        return self.distance_map

    def reset(self, agents: List[EnvAgent], rail: GridTransitionMap):
        """
        Reset the distance map
        """
        self.reset_was_called = True
        self.agents: List[EnvAgent] = agents
        self.rail = rail
        self.env_height = rail.height
        self.env_width = rail.width

    def _compute(self, agents: List[EnvAgent], rail: GridTransitionMap):
        self.agents_previous_computation = self.agents
        self.distance_map = np.inf * np.ones(shape=(len(agents),
                                                    self.env_height,
                                                    self.env_width,
                                                    4))
        for i, agent in enumerate(agents):
            self._distance_map_walker(rail, agent.target, i)

    def _distance_map_walker(self, rail: GridTransitionMap, position, target_nr: int):
        """
        Utility function to compute distance maps from each cell in the rail network (and each possible
        orientation within it) to each agent's target cell.
        """
        # Returns max distance to target, from the farthest away node, while filling in distance_map
        self.distance_map[target_nr, position[0], position[1], :] = 0

        # Fill in the (up to) 4 neighboring nodes
        # direction is the direction of movement, meaning that at least a possible orientation of an agent
        # in cell (row,col) allows a movement in direction `direction'
        nodes_queue = deque(self._get_and_update_neighbors(rail, position, target_nr, 0, enforce_target_direction=-1))

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
                valid_neighbors = self._get_and_update_neighbors(rail, (node[0], node[1]), target_nr, node[3], node[2])

                for n in valid_neighbors:
                    nodes_queue.append(n)

                if len(valid_neighbors) > 0:
                    max_distance = max(max_distance, node[3] + 1)

        return max_distance

    def _get_and_update_neighbors(self, rail: GridTransitionMap, position, target_nr, current_distance,
                                  enforce_target_direction=-1):
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
            new_cell = get_new_position(position, neigh_direction)

            if new_cell[0] >= 0 and new_cell[0] < self.env_height and new_cell[1] >= 0 and new_cell[1] < self.env_width:

                desired_movement_from_new_cell = (neigh_direction + 2) % 4

                # Check all possible transitions in new_cell
                for agent_orientation in range(4):
                    # Is a transition along movement `desired_movement_from_new_cell' to the current cell possible?
                    is_valid = rail.get_transition((new_cell[0], new_cell[1], agent_orientation),
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





def get_valid_move_actions_(agent_direction: Grid4TransitionsEnum,
                            agent_position: Tuple[int, int],
                            rail: GridTransitionMap) -> Set[RailEnvNextAction]:
    """
    Get the valid move actions (forward, left, right) for an agent.

    Parameters
    ----------
    agent_direction : Grid4TransitionsEnum
    agent_position: Tuple[int,int]
    rail : GridTransitionMap


    Returns
    -------
    Set of `RailEnvNextAction` (tuples of (action,position,direction))
        Possible move actions (forward,left,right) and the next position/direction they lead to.
        It is not checked that the next cell is free.
    """
    valid_actions: Set[RailEnvNextAction] = OrderedSet()
    possible_transitions = rail.get_transitions(*agent_position, agent_direction)
    num_transitions = np.count_nonzero(possible_transitions)
    # Start from the current orientation, and see which transitions are available;
    # organize them as [left, forward, right], relative to the current orientation
    # If only one transition is possible, the forward branch is aligned with it.
    if rail.is_dead_end(agent_position):
        action = RailEnvActions.MOVE_FORWARD
        exit_direction = (agent_direction + 2) % 4
        if possible_transitions[exit_direction]:
            new_position = get_new_position(agent_position, exit_direction)
            valid_actions.add(RailEnvNextAction(action, new_position, exit_direction))
    elif num_transitions == 1:
        action = RailEnvActions.MOVE_FORWARD
        for new_direction in [(agent_direction + i) % 4 for i in range(-1, 2)]:
            if possible_transitions[new_direction]:
                new_position = get_new_position(agent_position, new_direction)
                valid_actions.add(RailEnvNextAction(action, new_position, new_direction))
    else:
        for new_direction in [(agent_direction + i) % 4 for i in range(-1, 2)]:
            if possible_transitions[new_direction]:
                if new_direction == agent_direction:
                    action = RailEnvActions.MOVE_FORWARD
                elif new_direction == (agent_direction + 1) % 4:
                    action = RailEnvActions.MOVE_RIGHT
                elif new_direction == (agent_direction - 1) % 4:
                    action = RailEnvActions.MOVE_LEFT
                else:
                    raise Exception("Illegal state")

                new_position = get_new_position(agent_position, new_direction)
                valid_actions.add(RailEnvNextAction(action, new_position, new_direction))
    return valid_actions



def get_shortest_paths(distance_map: DistanceMap, max_depth: Optional[int] = None) \
    -> Dict[int, Optional[List[WalkingElement]]]:
    """
    Computes the shortest path for each agent to its target and the action to be taken to do so.
    The paths are derived from a `DistanceMap`.

    If there is no path (rail disconnected), the path is given as None.
    The agent state (moving or not) and its speed are not taken into account

    Parameters
    ----------
    distance_map

    Returns
    -------
        Dict[int, Optional[List[WalkingElement]]]

    """
    shortest_paths = dict()

    def _shortest_path_for_agent(agent):
        position = agent.position
        direction = agent.direction
        shortest_paths[agent.handle] = []
        distance = math.inf
        depth = 0
        while (position != agent.target and (max_depth is None or depth < max_depth)):
            next_actions = get_valid_move_actions_(direction, position, distance_map.rail)
            best_next_action = None
            for next_action in next_actions:
                next_action_distance = distance_map.get()[
                    agent.handle, next_action.next_position[0], next_action.next_position[
                        1], next_action.next_direction]
                if next_action_distance < distance:
                    best_next_action = next_action
                    distance = next_action_distance

            shortest_paths[agent.handle].append(WalkingElement(position, direction, best_next_action))
            depth += 1

            # if there is no way to continue, the rail must be disconnected!
            # (or distance map is incorrect)
            if best_next_action is None:
                shortest_paths[agent.handle] = None
                return

            position = best_next_action.next_position
            direction = best_next_action.next_direction
        if max_depth is None or depth < max_depth:
            shortest_paths[agent.handle].append(
                WalkingElement(position, direction,
                               RailEnvNextAction(RailEnvActions.STOP_MOVING, position, direction)))

    for agent in distance_map.agents:
        _shortest_path_for_agent(agent)

    return shortest_paths



class ShortestPathPredictorForRailEnv(PredictionBuilder):
    """
    ShortestPathPredictorForRailEnv object.

    This object returns shortest-path predictions for agents in the RailEnv environment.
    The prediction acts as if no other agent is in the environment and always takes the forward action.
    """

    def __init__(self, max_depth: int = 20):
        super().__init__(max_depth)

    def get(self, handle: int = None):
        """
        Called whenever get_many in the observation build is called.
        Requires distance_map to extract the shortest path.
        Does not take into account future positions of other agents!

        If there is no shortest path, the agent just stands still and stops moving.

        Parameters
        ----------
        handle : int, optional
            Handle of the agent for which to compute the observation vector.

        Returns
        -------
        np.array
            Returns a dictionary indexed by the agent handle and for each agent a vector of (max_depth + 1)x5 elements:
            - time_offset
            - position axis 0
            - position axis 1
            - direction
            - action taken to come here (not implemented yet)
            The prediction at 0 is the current position, direction etc.
        """
        agents = self.env.agents
        if handle:
            agents = [self.env.agents[handle]]

        distance_map = DistanceMap(self.env.agents, self.env.height, self.env.width, self.env.rail) 
               
        #distance_map: DistanceMap = self.env.distance_map
        shortest_paths = get_shortest_paths(distance_map, max_depth=self.max_depth)

        prediction_dict = {}
        for agent in agents:
            agent_virtual_position = agent.position
            

            agent_virtual_direction = agent.direction
            agent_speed = agent.speed_data["speed"]
            times_per_cell = int(np.reciprocal(agent_speed))
            prediction = np.zeros(shape=(self.max_depth + 1, 5))
            prediction[0] = [0, *agent_virtual_position, agent_virtual_direction, 0]

            shortest_path = shortest_paths[agent.handle]

            # if there is a shortest path, remove the initial position
            if shortest_path:
                shortest_path = shortest_path[1:]

            new_direction = agent_virtual_direction
            new_position = agent_virtual_position
            visited = OrderedSet()
            for index in range(1, self.max_depth + 1):
                # if we're at the target or not moving, stop moving until max_depth is reached
                if new_position == agent.target or not agent.moving or not shortest_path:
                    prediction[index] = [index, *new_position, new_direction, RailEnvActions.STOP_MOVING]
                    visited.add((*new_position, agent.direction))
                    continue

                if index % times_per_cell == 0:
                    new_position = shortest_path[0].position
                    new_direction = shortest_path[0].direction

                    shortest_path = shortest_path[1:]

                # prediction is ready
                prediction[index] = [index, *new_position, new_direction, 0]
                visited.add((*new_position, new_direction))

            # TODO: very bady side effects for visualization only: hand the dev_pred_dict back instead of setting on env!
            self.env.dev_pred_dict[agent.handle] = visited
            prediction_dict[agent.handle] = prediction

        return prediction_dict
