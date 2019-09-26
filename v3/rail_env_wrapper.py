
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator
from helper import *
from observation import RawObservation

class Rail_Env_Wrapper():

    def __init__(self, width=14, height=14, num_agents=2):
        self.num_agents = num_agents
        self.rail_gen = complex_rail_generator(
            nr_start_goal=3,
            nr_extra=3,
            min_dist=15,
            seed=random.randint(0,100000)
        )
        global_reward = 10
        self.done_last_step = {}
        self.dist = {}
        self.env = self.generate_env(width, height, global_reward, num_agents)
        self.num_of_done_agents = 0

    def step(self, actions):
        next_obs, rewards, done, _ = self.env.step(actions)
        next_obs = self.reshape_obs(next_obs)
        self.num_of_done_agents = self.modify_reward(rewards, done, self.done_last_step, self.num_of_done_agents, self.dist)
        self.done_last_step = done
        return next_obs ,done, rewards
    
    
    def reset(self):
        for i in range(self.num_agents):
            self.done_last_step[i] = 0
            self.dist[i] = 100
        obs = self.env.reset()
        return obs
        

    def modify_reward(self, rewards, done, done_last_step, num_of_done_agents, shortest_dist):
        for i in range(self.env.num_agents):
            if not done_last_step[i] and done[i]:
                num_of_done_agents += 1
                # Hand out some reward to all the agents
                for j in range(self.env.num_agents):
                    rewards[j] += 5  

                # Give some reward to our agent
                rewards[i] += 2**num_of_done_agents * 5
        
        for i in range(self.env.num_agents):
            agent = self.env.agents[i]
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

    #Eventuell auslagern nach Observations
    def reshape_obs(self, agent_observations):
        observations = []
        num_agents = len(agent_observations)
        for i in range(num_agents):
            agent_obs = agent_observations[i]
            observations.append(agent_obs)
        observations = np.array(observations)
        observations = np.swapaxes(observations,1,3)
        return observations

    def generate_env(self,
                    width,
                    height,
                    global_reward,
                    num_agents):
        self.env = RailEnv(
            width, 
            height, 
            self.rail_gen,
            schedule_generator = complex_schedule_generator(),
            number_of_agents=num_agents,
            obs_builder_object=RawObservation([21,21]))
        self.env.global_reward = global_reward
        self.env.num_agents = num_agents
        return self.env

    def get_agents_count(self):
        return self.env.num_agents

    def get_env(self):
        return self.env