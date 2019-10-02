import pandas as pd

from flatland.core.grid.grid4_astar import a_star
from flatland.envs.observations import (GlobalObsForRailEnv, LocalObsForRailEnv, TreeObsForRailEnv)
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import (complex_rail_generator, sparse_rail_generator)
from flatland.envs.schedule_generators import (random_schedule_generator, sparse_schedule_generator)

from flatland.utils.rendertools import RenderTool
from rail_env_wrapper import RailEnvWrapper
import constants

SEED = 5648

class RunStatistics:
    def __init__(self,num_agents,num_agents_done,total_reward,width,height,steps_needed,evaluation_round):
        self.num_agents = num_agents
        self.num_agents_done = num_agents_done
        self.total_reward = total_reward
        self.width = width
        self.height = height
        self.evaluation_round = evaluation_round
        self.steps_needed = steps_needed

    def to_dict(self):
        return {
            'num_agents' : self.num_agents,
            'num_agents_done' : self.num_agents_done,
            'total_reward' : self.total_reward,
            'width' : self.width,
            'height' : self.height,
            'evaluation_round' : self.evaluation_round,
            'steps_needed' : self.steps_needed
        }

    def __str__(self):
        return str(self.to_dict())

class Evaluator:
    def __init__(self):
        self.stats = []
        self.num_agents = 1
        self.grid_width = 20
        self.grid_height = 20
        self.env = RailEnvWrapper()

    def set_benchmark_submission(self,benchmark_submission):
        """ set model function actions = func(observations,num_agents)
        """
        self.get_policy = benchmark_submission['get_policy']
        self.observation = benchmark_submission['observation']

    def sum_up_dict(self,dict, num_agents):
        val = 0
        for i in range(num_agents):
            val += dict[i]
        return val

    def count_agents_done(self,done_dict, num_agents):
        agents_done = 0
        for i in range(num_agents):
            if done_dict[i] == True:
                agents_done += 1
        
        return agents_done

    def run_to_end(self, initial_obs, num_agents):
        done = False
        step = 0
        total_reward = 0
        obs = initial_obs

        while not done and step < self.env.max_steps:
            actions = self.get_policy(obs, self.num_agents)
            obs, rewards, agents_done = self.env.step(actions)
            done = agents_done['__all__']
            total_reward += self.sum_up_dict(rewards, num_agents)
            step += 1

        num_agents_done = self.count_agents_done(agents_done,num_agents)
        return num_agents_done,total_reward, step

    def save_stats(self, num_agents_done, total_reward, steps_needed, round):
        run_stats = RunStatistics(
                        self.num_agents,
                        num_agents_done,
                        total_reward,
                        self.grid_width,
                        self.grid_height,
                        steps_needed,
                        round
                    )

        print(run_stats)
        self.stats.append(run_stats)

    def change_grid_round1(self):
        self.env.update_env_with_params(
            width=12,
            height=12,
            num_agents=1,
            max_steps = 40,
            rail_type = 'complex',
            rail_gen_params = {
                'nr_start_goal': 3,
                'nr_extra': 3,
                'min_dist': 12,
                'max_dist' : 99999
            },
            seed=SEED
        )


    def change_grid_round2(self):
        self.env.update_env_with_params(
            width=20,
            height=20,
            num_agents=1,
            max_steps = 40,
            rail_type = 'complex',
            rail_gen_params = {
                'nr_start_goal': 6,
                'nr_extra': 6,
                'min_dist': 12,
                'max_dist' : 99999
            },
            seed=SEED
        )
    
    def change_grid_round3(self):
        self.env.update_env_with_params(
            width=20,
            height=20,
            num_agents=2,
            max_steps = 40,
            rail_type = 'complex',
            rail_gen_params = {
                'nr_start_goal': 6,
                'nr_extra': 6,
                'min_dist': 12,
                'max_dist' : 99999
            },
            seed=SEED
        )

    def change_grid_round4(self):
        self.env.update_env_with_params(
            width=20,
            height=20,
            num_agents=2,
            max_steps = 40,
            rail_type = 'sparse',
            rail_gen_params = {
                'num_cities': 10,  # Number of cities in map
                'num_intersections': 10,  # Number of interesections in map
                'num_trainstations': 20,  # Number of possible start/targets on map
                'min_node_dist': 6,  # Minimal distance of nodes
                'node_radius': 3, # Proximity of stations to city center
                'num_neighb': 3 # Number of connections to other cities
            },  
            seed=SEED  
        )

    def change_grid_round5(self):
        self.env.update_env_with_params(
            width=60,
            height=60,
            num_agents=6,
            max_steps = 40,
            rail_type = 'sparse',
            rail_gen_params = {
                'num_cities': 10,  # Number of cities in map
                'num_intersections': 30,  # Number of interesections in map
                'num_trainstations': 12,  # Number of possible start/targets on map
                'min_node_dist': 6,  # Minimal distance of nodes
                'node_radius': 3, # Proximity of stations to city center
                'num_neighb': 5 # Number of connections to other cities
            },  
            seed=SEED
        )

    def run_episodes(self, episode_no, num_episodes):
        for r in range(num_episodes):
            obs = self.env.reset()
            num_agents_done, total_reward, steps_needed = self.run_to_end(obs, self.num_agents)
            self.save_stats(num_agents_done, total_reward, steps_needed, episode_no)

    def start_evaluation(self):
        print('Round 1 - simple environment with one agent')
        self.change_grid_round1()
        self.run_episodes(1, 20)

        print('Round 2 - Single agent, more difficult environment')
        self.change_grid_round2()
        self.run_episodes(2, 20)

        print('Round 3 - Two agents, more difficult environment')
        self.change_grid_round3()
        self.run_episodes(3, 20)

        print('Round 4 - four agents, even more difficult environment')
        self.change_grid_round4()
        self.run_episodes(4, 20)
        
        print('Round 5 - Evaluation environment')
        self.change_grid_round5()
        self.run_episodes(5, 20)


    def save_stats_to_csv(self,run_name):
        df = pd.DataFrame.from_records([s.to_dict() for s in self.stats])
        df.to_csv(run_name + '_report.csv')