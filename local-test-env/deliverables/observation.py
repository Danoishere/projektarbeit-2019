
import numpy as np
import deliverables.input_params as params
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

class RailObsBuilder(TreeObsForRailEnv):
    def __init__(self):
        super().__init__(params.tree_depth, ShortestPathPredictorForRailEnv(40))
        self.last_obs = {}

    def get_many(self, handles=None):
        obs = super().get_many(handles=handles)
        all_augmented_obs = {}

        for handle in obs:
            last_agent_obs = None
            if handle in self.last_obs:
                last_agent_obs = self.last_obs[handle]

            next_agent_obs = obs[handle]
            next_agent_tree_obs, vec_obs = self.reshape_agent_obs(handle, next_agent_obs, None)
            augmented_tree_obs = self.augment_agent_tree_obs_with_frames(last_agent_obs, next_agent_tree_obs)
            all_augmented_obs[handle] = (augmented_tree_obs, vec_obs)

        return all_augmented_obs

    def reshape_agent_obs(self, handle, agent_obs, info):
        if agent_obs is None:
            # New tree-obs is just the size of one frame
            tree_obs = np.zeros(params.frame_size)
            vec_obs = np.zeros(params.vec_state_size)
            return tree_obs.astype(np.float32), vec_obs.astype(np.float32)

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
            for layer in obs_layers:
                for node in layer:
                    node_obs = node_to_obs(node)
                    tree_obs.append(node_obs)
                    
            tree_obs = np.concatenate(tree_obs)

            agent = self.env.agents[handle]

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

            return tree_obs.astype(np.float32), vec_obs.astype(np.float32)


    def augment_agent_tree_obs_with_frames(self, last_obs, next_obs):
        single_obs_len = params.frame_size
        full_obs_len = params.tree_state_size

        if last_obs == None:
            last_obs = np.zeros(full_obs_len)

        last_multi_frame_obs = last_obs[:single_obs_len]
        multi_frame_obs = np.zeros(full_obs_len)

        # Start = new obs
        multi_frame_obs[:single_obs_len] = next_obs

        # Fill remaining n-1 slots with last obs
        multi_frame_obs[single_obs_len:single_obs_len+len(last_multi_frame_obs)] = last_multi_frame_obs
        return multi_frame_obs.astype(np.float32)


def buffer_to_obs_lists(episode_buffer):
    tree_obs = np.asarray([row[0][0] for row in episode_buffer])
    vec_obs = np.asarray([row[0][1] for row in episode_buffer])
    return tree_obs, vec_obs


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
        return 0.1 + field/norm_val
    
    

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
        dir_num,
        normalize_field(node.dist_min_to_target),
        normalize_field(node.dist_other_agent_encountered),
        normalize_field(node.dist_other_target_encountered),
        normalize_field(node.dist_own_target_encountered),
        normalize_field(node.dist_potential_conflict),
        normalize_field(node.dist_to_next_branch),
        normalize_field(node.dist_unusable_switch),
        normalize_field(node.num_agents_malfunctioning,10),
        normalize_field(node.num_agents_opposite_direction, 10),
        normalize_field(node.num_agents_ready_to_depart, 20),
        normalize_field(node.num_agents_same_direction, 20),
        node.speed_min_fractional,
        dist_left,
        dist_right,
        dist_forward
    ]

    return obs