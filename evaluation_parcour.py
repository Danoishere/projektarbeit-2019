import constants

from observation import RawObservation
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool
from flatland.envs.observations import TreeObsForRailEnv, LocalObsForRailEnv, GlobalObsForRailEnv
from flatland.envs.rail_generators import complex_rail_generator, sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator, complex_schedule_generator
from flatland.core.grid.grid4_astar import a_star

SEED = 5648

class RunStatistics:
    def __init__(self,num_agents,agents_done,total_reward,nr_start_goal,nr_extra,grid_width,grid_height,evaluation_round):
        self.num_agents = num_agents
        self.agents_done = agents_done
        self.total_reward = total_reward
        self.nr_start_goal = nr_start_goal
        self.nr_extra = nr_extra
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.evaluation_round = evaluation_round

    def to_dict(self):
        return {
            'num_agents' : self.num_agents,
            'agents_done' : self.agents_done,
            'total_reward' : self.total_reward,
            'nr_start_goal' : self.nr_start_goal,
            'nr_extra' : self.nr_extra,
            'grid_width' : self.grid_width,
            'grid_height' : self.grid_height,
            'evaluation_round' : self.evaluation_round
        }

class Evaluator:
    def __init__(self):
        self.stats = []
        self.nr_start_goal = 1
        self.nr_extra = 1
        self.num_agents = 1
        self.grid_width = 10
        self.grid_height = 10
        self.rail_generator = complex_rail_generator(
                                            nr_start_goal=self.nr_start_goal,
                                            nr_extra=self.nr_extra,
                                            min_dist=5,
                                            max_dist=99999,
                                            seed=SEED)

        self.schedule_generator = complex_schedule_generator()
        
    def create_env(self):
        env = RailEnv(
                    width=self.grid_width,
                    height=self.grid_height,
                    rail_generator = self.rail_generator,
                    schedule_generator = self.schedule_generator,
                    number_of_agents=self.num_agents,
                    obs_builder_object=RawObservation([20,20]))

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

        num_agents_done = self.count_agents_done(agents_done,num_agents)

        return num_agents_done,total_reward

    def save_stats(self, num_agents_done, total_reward, round):
        run_stats = RunStatistics(
                self.num_agents,
                num_agents_done,
                total_reward,
                self.nr_start_goal,
                self.nr_extra,
                self.grid_width,
                self.grid_height,
                round
                )
        self.stats.append(run_stats)


    def change_grid_round2(self):
        self.num_agents = 2
        self.grid_width = 20
        self.grid_height = 20
        self.nr_start_goal = 4
        self.nr_extra = 4
        self.env = self.create_env()

    def change_grid_round3(self):
        self.rail_generator=sparse_rail_generator(
            num_cities=10,  # Number of cities in map
            num_intersections=10,  # Number of interesections in map
            num_trainstations=50,  # Number of possible start/targets on map
            min_node_dist=6,  # Minimal distance of nodes
            node_radius=3,  # Proximity of stations to city center
            num_neighb=3,  # Number of connections to other cities
            seed=5,  # Random seed
            grid_mode=False  # Ordered distribution of nodes
        )
        self.schedule_generator = sparse_schedule_generator()
        self.env = self.create_env()


    def start_evaluation(self):
        self.env = self.create_env()
        # Round 1 - simple environment with one agent
        for r in range(25):
            obs = self.env.reset()
            num_agents_done, total_reward = self.run_to_end(obs, self.num_agents)
            self.save_stats(num_agents_done,total_reward,1)

        # Round 2 - Two agents, more difficult environment
        self.change_grid_round2()
        for r in range(25):
            obs = self.env.reset()
            num_agents_done, total_reward = self.run_to_end(obs, self.num_agents)
            self.save_stats(num_agents_done,total_reward,2)

        # Round 3 - four agents, even more difficult environment
        self.change_grid_round3()
        for r in range(25):
            obs = self.env.reset()
            num_agents_done, total_reward = self.run_to_end(obs, self.num_agents)
            self.save_stats(num_agents_done,total_reward,3)

        # Round 4 - Evaluation environment
        self.change_grid_round4()
        for r in range(25):
            obs = self.env.reset()
            num_agents_done, total_reward = self.run_to_end(obs, self.num_agents)
            self.save_stats(num_agents_done,total_reward,4)

        

        

            

            
            


            





