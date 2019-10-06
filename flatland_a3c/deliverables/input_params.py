max_episode_length = 40
gamma = 0.99
entropy_factor = 0.03
learning_rate = 1e-4
# gradient_norm_actor = 4.0
# gradient_norm_critic = 7.0

gradient_norm_actor = 8.0
gradient_norm_critic = 14.0

number_of_inputs = 4
number_of_actions = 5
map_state_size = (11,11,9) 
grid_state_size = (11,11,16,1)
vector_state_size = (5,)
tree_state_size = (231,)
