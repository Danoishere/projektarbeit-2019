max_episode_length = 40
gamma = 0.99
entropy_factor = 0.05
learning_rate = 1e-5

gradient_norm = 1.0

number_of_actions = 5
number_of_comm = 5
num_of_obs = 2

tree_depth = 3
path_root = tree_depth + 1
path_sec = tree_depth
path_thrd = tree_depth - 1

recurrent_size = 64

num_features = 34

tree_state_size = 0
for d in range(tree_depth):
    tree_state_size += 2**d

tree_state_size *= num_features
vec_state_size = 11

tot_obs_size = tree_state_size + vec_state_size
print(tot_obs_size)
