max_episode_length = 40
gamma = 0.99
entropy_factor = 0.01
learning_rate = 1e-4

gradient_norm = 15.0

number_of_inputs = 4
number_of_actions = 5

tree_depth = 5
feature_branches = 5
num_features = 13
num_frames = 3

# Root-node + n branches on m layers * l features + n features for later usage
frame_size = (tree_depth+1)*feature_branches*num_features + num_features
vec_tree_state_size = frame_size*num_frames
