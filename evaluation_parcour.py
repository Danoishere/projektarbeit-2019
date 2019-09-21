import pandas as pd

from flatland.core.grid.grid4_astar import a_star
from flatland.envs.observations import (GlobalObsForRailEnv,
                                        LocalObsForRailEnv, TreeObsForRailEnv)
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import (complex_rail_generator,
                                           sparse_rail_generator)
from flatland.envs.schedule_generators import (random_schedule_generator,
                                               sparse_schedule_generator)
from flatland.utils.rendertools import RenderTool

import constants
from observation import RawObservation

SEED = 5648

class RunStatistics:
    def __init__(self,num_agents,agents_done,total_reward,nr_start_goal,nr_extra,grid_width,grid_height,steps_needed,evaluation_round):
        self.num_agents = num_agents
        self.agents_done = agents_done
        self.total_reward = total_reward
        self.nr_start_goal = nr_start_goal
        self.nr_extra = nr_extra
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.evaluation_round = evaluation_round
        self.steps_needed = steps_needed

    def to_dict(self):
        return {
            'num_agents' : self.num_agents,
            'agents_done' : self.agents_done,
            'total_reward' : self.total_reward,
            'nr_start_goal' : self.nr_start_goal,
            'nr_extra' : self.nr_extra,
            'grid_width' : self.grid_width,
            'grid_height' : self.grid_height,
            'evaluation_round' : self.evaluation_round,
            'steps_needed' : self.steps_needed
        }

    def __str__(self):
        return str(self.to_dict())

class Evaluator:
    def __init__(self):
        self.stats = []
        self.nr_start_goal = 2
        self.nr_extra = 1
        self.num_agents = 1
        self.grid_width = 20
        self.grid_height = 20
        self.rail_generator = complex_rail_generator(
                                            nr_start_goal=self.nr_start_goal,
                                            nr_extra=self.nr_extra,
                                            min_dist=2,
                                            max_dist=99999,
                                            seed=SEED)

        self.schedule_generator = random_schedule_generator()
        
    def create_env(self):
        env = RailEnv(
                    width=self.grid_width,
                    height=self.grid_height,
                    rail_generator = self.rail_generator,
                    schedule_generator = self.schedule_generator,
                    number_of_agents=self.num_agents,
                    obs_builder_object=RawObservation([21,21]))

        return env

    def set_model(self,model_function):
        """ set model function actions = func(observations,num_agents)
        """
        self.model_function = model_function

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
        MAX_STEPS_PER_EPISODE = 200
        done = False
        step = 0
        total_reward = 0
        obs = initial_obs

        while not done and step < MAX_STEPS_PER_EPISODE:
            actions = self.model_function(obs, self.num_agents)
            obs, rewards, agents_done, _ = self.env.step(actions)
            done = agents_done['__all__']
            total_reward += self.sum_up_dict(rewards, num_agents)
            step += 1

        num_agents_done = self.count_agents_done(agents_done,num_agents)

        return num_agents_done,total_reward,step

    def save_stats(self, num_agents_done, total_reward, steps_needed, round):
        run_stats = RunStatistics(
                self.num_agents,
                num_agents_done,
                total_reward,
                self.nr_start_goal,
                self.nr_extra,
                self.grid_width,
                self.grid_height,
                steps_needed,
                round
                )

        print(run_stats)
        self.stats.append(run_stats)


    def change_grid_round2(self):
        self.num_agents = 1
        self.grid_width = 20
        self.grid_height = 20
        self.nr_start_goal = 6
        self.nr_extra = 6

        self.rail_generator = complex_rail_generator(
                                            nr_start_goal=self.nr_start_goal,
                                            nr_extra=self.nr_extra,
                                            min_dist=2,
                                            max_dist=99999,
                                            seed=SEED)

        self.env = self.create_env()
    
    def change_grid_round3(self):
        self.num_agents = 2
        self.grid_width = 20
        self.grid_height = 20
        self.nr_start_goal = 6
        self.nr_extra = 6

        self.rail_generator = complex_rail_generator(
                                            nr_start_goal=self.nr_start_goal,
                                            nr_extra=self.nr_extra,
                                            min_dist=2,
                                            max_dist=99999,
                                            seed=SEED)
                                            
        self.env = self.create_env()

    def change_grid_round4(self):
        self.num_agents = 4
        self.grid_width = 20
        self.grid_height = 20

        self.rail_generator=sparse_rail_generator(
            num_cities=10,  # Number of cities in map
            num_intersections=10,  # Number of interesections in map
            num_trainstations=20,  # Number of possible start/targets on map
            min_node_dist=6,  # Minimal distance of nodes
            node_radius=3,  # Proximity of stations to city center
            num_neighb=3,  # Number of connections to other cities
            seed=5,  # Random seed
            grid_mode=False  # Ordered distribution of nodes
        )
        self.schedule_generator = sparse_schedule_generator()
        self.env = self.create_env()

    def change_grid_round5(self):
        self.num_agents = 6
        self.grid_width = 60
        self.grid_height = 60
        self.rail_generator=sparse_rail_generator(
            num_cities=10,  # Number of cities in map
            num_intersections=30,  # Number of interesections in map
            num_trainstations=12,  # Number of possible start/targets on map
            min_node_dist=6,  # Minimal distance of nodes
            node_radius=3,  # Proximity of stations to city center
            num_neighb=5,  # Number of connections to other cities
            seed=5,  # Random seed
            grid_mode=False  # Ordered distribution of nodes
        )
        self.env = self.create_env()

    def run_episodes(self, episode_no, num_episodes):
        for r in range(num_episodes):
            obs = self.env.reset()
            num_agents_done, total_reward,steps_needed = self.run_to_end(obs, self.num_agents)
            self.save_stats(num_agents_done,total_reward,steps_needed,episode_no)

    def start_evaluation(self):
        self.env = self.create_env()
        
        print('Round 1 - simple environment with one agent')
        self.run_episodes(1, 20)

        print('Round 2 - Single agent, more difficult environment')
        self.change_grid_round2()
        self.run_episodes(2, 16)

        print('Round 3 - Two agents, more difficult environment')
        self.change_grid_round3()
        self.run_episodes(3, 16)

        """
        print('Round 4 - four agents, even more difficult environment')
        self.change_grid_round4()
        self.run_episodes(4, 5)
        
        print('Round 5 - Evaluation environment')
        self.change_grid_round5()
        self.run_episodes(5, 4)
        """


    def analyze_stats(self,run_name):
        df = pd.DataFrame.from_records([s.to_dict() for s in self.stats])
        df.to_csv(run_name + '_report.csv')
