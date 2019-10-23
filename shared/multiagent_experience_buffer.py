import pandas as pd


class MultiagentExperienceBuffer:
    def __init__(self):
        self.df = pd.DataFrame(columns=['agent_id','observation','action','next_observation','reward', 'agent_done'])

    def save_observations(self, observations, actions, rewards):
        for i in range(self.env.num_agents):
            agent_obs = self.obs_helper.obs_for_agent(observations, i) 
            agent_action = actions[i]
            agent_reward = rewards[i]
            agent_next_obs =  self.obs_helper.obs_for_agent(next_obs, i) 

            if not done_last_step[i]:
                episode_buffer[i].append([
                    agent_obs,
                    agent_action,
                    agent_reward,
                    agent_next_obs,
                    episode_done,
                    0])


    def add(self, agent_id, agent_obs, agent_action, agent_reward, agent_next_obs, agent_done, episode_done):
        self.df = self.df.append({
            'agent_id' : agent_id,
            'observation' : agent_obs,
            'action' : agent_action,
            'next_observation':agent_next_obs,
            'reward' : agent_reward,
            'agent_done' : agent_done
        }, ignore_index=True)


    