max_episode_length = 40
gamma = 0.99
entropy_factor = 0.0
learning_rate = 1e-4

gradient_norm = 15.0

number_of_actions = 5
num_of_obs = 2

tree_depth = 5
path_root = tree_depth + 1
path_sec = tree_depth
path_thrd = tree_depth - 1

num_features = 16
num_frames = 3

# Root-node + n branches on m layers * l features + n features for later usage
frame_size = (path_root + path_sec + path_thrd)*num_features
tree_state_size = frame_size*num_frames

vec_state_size = 10
