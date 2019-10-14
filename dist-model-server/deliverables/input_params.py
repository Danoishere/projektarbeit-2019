max_episode_length = 40
gamma = 0.996
entropy_factor = 0.001
learning_rate = 1e-4

gradient_norm = 15.0

number_of_inputs = 4
number_of_actions = 5
#map_state_size = (11,11,7) 
#grid_state_size = (11,11,1)

tree_depth = 6
feature_branches = 5
num_features = 13

# Root-node + n branches on m layers * l features + n features for later usage
vec_tree_state_size = ((tree_depth+1)*feature_branches*num_features + num_features,)
