max_episode_length = 40
gamma = 0.99
entropy_factor = 0.05
learning_rate = 1e-4

gradient_norm = 1.0

number_of_actions = 5
number_of_comm = 5
num_of_obs = 2

tree_depth = 3
path_root = tree_depth + 1
path_sec = tree_depth
path_thrd = tree_depth - 1

recurrent_size = 64

num_features = 29
num_frames = 1

# Root-node + n branches on m layers * l features + n features for later usage
frame_size = (path_root + path_sec + path_thrd)*num_features
comm_size = (path_root + path_sec + path_thrd)*number_of_comm
tree_state_size = frame_size*num_frames
vec_state_size = 10

tot_obs_size = tree_state_size + vec_state_size + comm_size
