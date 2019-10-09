def modify_reward(rewards, env, done, done_last_step, num_of_done_agents, shortest_dist):
    for i in range(env.num_agents):
        if not done_last_step[i] and done[i]:
            num_of_done_agents += 1
            # Hand out some reward to all the agents (collaboration reward)
            for j in range(env.num_agents):
                rewards[j] += 10

            # Give some reward to our agent
            #rewards[i] += 2**num_of_done_agents * 5
    
    return num_of_done_agents
