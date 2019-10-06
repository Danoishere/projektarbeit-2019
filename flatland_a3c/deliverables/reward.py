def modify_reward(rewards, env, done, done_last_step, num_of_done_agents, shortest_dist):
    
    for i in range(env.num_agents):
        if not done_last_step[i] and done[i]:
            num_of_done_agents += 1
            # Hand out some reward to all the agents
            for j in range(env.num_agents):
                rewards[j] += 5  

            # Give some reward to our agent
            rewards[i] += 2**num_of_done_agents * 5
    
    '''
    for i in range(env.num_agents):
        agent = env.agents[i]
        path_to_target = agent.path_to_target
        current_path_length = len(path_to_target)
        shortest_path_length = shortest_dist[i]

    # Adding reward for getting closer to goal
    if current_path_length < shortest_path_length:
        rewards[i] +=1
        shortest_dist[i] = current_path_length

    # Subtract reward for getting further away
    if current_path_length > shortest_path_length:
        rewards[i] -= 1
    '''
        
    return num_of_done_agents