from dill import copy
import numpy as np

map_state_size = (11,11,9)
grid_state_size = (11,11,16)
vector_state_size = 5

def run_attempt(obs, model, env, max_attempt_length):
        action_list = []
        attempt_step_count = 0
        attempt_reward = 0
        num_of_done_agents = 0
        episode_done = False
        env_copy = copy(env)

        done_last_step = {i: False for i in range(env.num_agents)}                
        dist = {i: 100 for i in range(env.num_agents)}
        
        while not episode_done and attempt_step_count < max_attempt_length:
            #Take an action using probabilities from policy network output.
            predcition = model.predict([obs[0],obs[1],obs[2]])
            actions = {}
            for i in range(env_copy.num_agents):
                a_dist = predcition[0][i]
                a = np.random.choice([0,1,2,3,4], p = a_dist)
                actions[i] = a

            action_list.append(actions)
            next_obs, rewards, done, _ = env_copy.step(actions)
            next_obs = reshape_obs(next_obs)
            num_of_done_agents = modify_reward(env_copy, rewards, done, done_last_step, num_of_done_agents, dist)

            for i in range(env.num_agents):
                if not done_last_step[i]:
                    attempt_reward += rewards[i]

            # Is episode finished?
            episode_done = done['__all__']

            attempt_step_count += 1
            done_last_step = done
            obs = next_obs
            
        if not episode_done:
            # Bootstrap with value-function for remaining reward
            for i in range(env.num_agents):
                if not done_last_step[i]:
                    attempt_reward += rewards[i] + predcition[1][i]

        return episode_done, attempt_reward, action_list, attempt_step_count

def reshape_obs(agent_observations):
    import numpy as np
    map_obs = []
    vec_obs = []
    grid_obs = []
    num_agents = len(agent_observations)

    for i in range(num_agents):
        agent_obs = agent_observations[i]
        map_obs.append(agent_obs[0])
        grid_obs.append(agent_obs[1])
        vec_obs.append(agent_obs[2])
        
    map_obs = np.asarray(map_obs).astype(np.float32)
    map_obs = np.reshape(map_obs,(num_agents, map_state_size[0],map_state_size[1],map_state_size[2]))

    grid_obs = np.asarray(grid_obs).astype(np.float32)
    grid_obs = np.reshape(grid_obs,(num_agents, grid_state_size[0],grid_state_size[1],grid_state_size[2],1))

    vec_obs = np.asarray(vec_obs).astype(np.float32)
    return [map_obs, grid_obs, vec_obs]

def modify_reward(env, rewards, done, done_last_step, num_of_done_agents, shortest_dist):
    for i in range(env.num_agents):
        if not done_last_step[i] and done[i]:
            num_of_done_agents += 1
            # Hand out some reward to all the agents
            for j in range(env.num_agents):
                rewards[j] += 5  

            # Give some reward to our agent
            rewards[i] += 2**num_of_done_agents * 5

    
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
    
    return num_of_done_agents