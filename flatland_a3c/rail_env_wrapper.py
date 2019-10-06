
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import complex_rail_generator, sparse_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator, sparse_schedule_generator, random_schedule_generator
import random

from deliverables.reward import modify_reward

class RailEnvWrapper():
    initial_step_penalty = -2
    global_reward = 10

    def __init__(self, observation_builder, width=14, height=14, num_agents=2):
        self.num_agents = num_agents
        
        self.schedule_gen = random_schedule_generator()
        self.rail_gen = complex_rail_generator(
            nr_start_goal=6,
            nr_extra=6,
            min_dist=10,
            seed=random.randint(0,100000)
        )

        self.done_last_step = {}
        self.observation_builder = observation_builder
        self.dist = {}

        self.env = self.generate_env(width, height)
        self.num_of_done_agents = 0
        self.episode_step_count = 0
        self.max_steps = 40

    def step(self, actions):
        self.env.step_penalty = -2*1.02**self.episode_step_count
        next_obs, rewards, done, _ = self.env.step(actions)
        self.num_of_done_agents = modify_reward(rewards, self.env, done, self.done_last_step, self.num_of_done_agents, self.dist)
        self.done_last_step = done
        return next_obs, rewards, done
    
    
    def reset(self):
        for i in range(self.num_agents):
            self.done_last_step[i] = False
            self.dist[i] = 100
            
        obs = self.env.reset()
        while obs[0].shape[0] == 0:
            obs = self.env.reset()

        self.env.step_penalty = self.initial_step_penalty
        self.episode_step_count = 0
        return obs


    def generate_env(self, width, height):
        self.width = width
        self.height = height
        self.env = RailEnv(
            width, 
            height, 
            self.rail_gen,
            schedule_generator = self.schedule_gen,
            number_of_agents=self.num_agents,
            obs_builder_object=self.observation_builder)

        self.env.global_reward = self.global_reward
        self.env.num_agents = self.num_agents
        self.env.step_penalty = -2
        return self.env

    def get_agents_count(self):
        return self.env.num_agents

    def get_env(self):
        return self.env

    def update_env_with_params(self, width, height, num_agents, max_steps, rail_type, rail_gen_params, seed=-1):
        if seed == -1:
            seed=random.randint(0,100000)

        self.num_agents = num_agents
        self.max_steps = max_steps

        if rail_type == 'complex':
            self.rail_gen = complex_rail_generator(
                nr_start_goal=rail_gen_params['nr_start_goal'],
                nr_extra=rail_gen_params['nr_extra'],
                min_dist=rail_gen_params['min_dist'],
                max_dist=rail_gen_params['max_dist'],
                seed=seed
            )

            self.schedule_gen = complex_schedule_generator()
        elif rail_type == 'sparse':
            self.rail_gen = sparse_rail_generator(
                num_cities=rail_gen_params['num_cities'],
                num_intersections=rail_gen_params['num_intersections'],
                num_trainstations=rail_gen_params['num_trainstations'],
                min_node_dist=rail_gen_params['min_node_dist'],
                node_radius=rail_gen_params['node_radius'],
                num_neighb=rail_gen_params['num_neighb'],
                grid_mode=rail_gen_params['grid_mode'],
                enhance_intersection=rail_gen_params['enhance_intersection'],
                seed=seed)

            self.schedule_gen = sparse_schedule_generator()
        else:
            raise ValueError('Please specify either "complex" or "sparse" as rail_type')

        self.generate_env(width, height)
            

