
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator, complex_rail_generator 
from flatland.envs.schedule_generators import sparse_schedule_generator
import random




class RailEnvWrapper():
    initial_step_penalty = -2

    def __init__(self, observation_builder, width=12, height=12, num_agents=2):
        self.num_agents = num_agents
        
        self.schedule_gen = sparse_schedule_generator({   
                    1.: 0.25,       # Fast passenger train
                    1. / 2.: 0.25,  # Fast freight train
                    1. / 3.: 0.25,  # Slow commuter train
                    1. / 4.: 0.25
                })

        self.stochastic_data = {
                'prop_malfunction': 0.0,  # Percentage of defective agents
                'malfunction_rate': 30,  # Rate of malfunction occurence
                'min_duration': 3,  # Minimal duration of malfunction
                'max_duration': 20  # Max duration of malfunction
        }
        

        self.done_last_step = {}
        self.observation_builder = observation_builder
        self.dist = {}

        self.num_of_done_agents = 0
        self.episode_step_count = 0
        self.max_steps = 40
        self.update_env_with_params(
            width=30,
            height=30,
            num_agents=1,
            max_steps = 100,
            rail_type = 'sparse',
            rail_gen_params = {
                'num_cities': 2,
                'grid_mode': False,
                'max_rails_between_cities': 1,
                'max_rails_in_city' : 2
            }
        )

    def step(self, actions):
        self.env.step_penalty = -2*1.02**self.episode_step_count
        next_obs, rewards, done, _ = self.env.step(actions)
        self.done_last_step = dict(done)
        self.episode_step_count += 1
        return next_obs, rewards, done
    
    
    def reset(self):
        for i in range(self.num_agents):
            self.done_last_step[i] = False
            self.dist[i] = 100
        
        obs = self.env.reset()
        
        # Obs-shape must be equal to num of agents, otherwise, the level couldn't be generated orderly
        while len(obs[0]) != self.num_agents:
            obs = self.env.reset()

        self.num_of_done_agents = 0
        self.env.step_penalty = self.initial_step_penalty
        self.episode_step_count = 0
        return obs


    def generate_env(self, width, height):
        self.width = width
        self.height = height
        self.env = RailEnv(
            width, 
            height, 
            rail_generator=self.rail_gen,
            stochastic_data=self.stochastic_data,
            schedule_generator = self.schedule_gen,
            number_of_agents=self.num_agents,
            obs_builder_object=self.observation_builder,
            remove_agents_at_target=True)

        #self.env.global_reward = self.global_reward
        self.env.num_agents = self.num_agents
        self.env.step_penalty = self.initial_step_penalty
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

            #self.schedule_gen = complex_schedule_generator()
        elif rail_type == 'sparse':
            self.rail_gen = sparse_rail_generator(
                max_num_cities=rail_gen_params['num_cities'],
                seed=seed,
                grid_mode=rail_gen_params['grid_mode'],
                max_rails_between_cities=rail_gen_params['max_rails_between_cities'],
                max_rails_in_city=rail_gen_params['max_rails_in_city']
            )
            
        else:
            raise ValueError('Please specify either "complex" or "sparse" as rail_type')

        self.generate_env(width, height)
            

