import constant as const
import requests
from random import randint, random

class Curriculum:
    def __init__(self):
        # Seed = -1 means create random seed
        self.seed = -1
        self.randomize_level_generation = False
        self.curriculum = {
            0: {
                'level_generator' : lambda env: self.change_grid_round0(env),
                'switch_on_successrate': 0.9
                },
            1: {
                'level_generator' : lambda env: self.change_grid_round1(env),
                'switch_on_successrate': 0.9
                },
            2: {
                'level_generator' : lambda env: self.change_grid_round2(env),
                'switch_on_successrate': 0.9
                },
            3: {
                'level_generator' : lambda env: self.change_grid_round3(env),
                'switch_on_successrate': 0.9
                },
            4: {
                'level_generator' : lambda env: self.change_grid_round4(env),
                'switch_on_successrate': 0.9
                },
            5: {
                'level_generator' : lambda env: self.change_grid_round5(env),
                'switch_on_successrate': 1.0
                }
        }

        self.number_of_levels = len(self.curriculum)
        self.update_curriculum_level()
        
    def should_switch_level(self, successrate):
        switch_on_successrate = self.curriculum[self.current_level]['switch_on_successrate']
        return successrate >= switch_on_successrate
            

    def update_env_to_curriculum_level(self, env):
        env_level = self.current_level
        if self.randomize_level_generation:
            # Take the new level with a higher probability
            if random() >= 0.0:
                env_level = self.current_level
            else:  
                env_level = randint(0, self.current_level)
        
        self.active_level = env_level
        return self.curriculum[env_level]['level_generator'](env)


    def update_curriculum_level(self):
        resp = requests.get(url=const.url + '/curriculum_level')
        self.current_level = resp.json()['curriculum_lvl']


    def increase_curriculum_level(self):
        resp = requests.get(url=const.url + '/increase_curriculum_level')
        self.update_curriculum_level()
        print('Curriculum level increased')

    def change_grid_round0(self, env):
        env.update_env_with_params(
            width=30,
            height=30,
            num_agents=2,
            max_steps = 180,
            rail_type = 'sparse',
            rail_gen_params = {
                'num_cities': 2,
                'grid_mode': False,
                'max_rails_between_cities': 1,
                'max_rails_in_city' : 2
            },
            seed = self.seed        
        )

    def change_grid_round1(self, env):
        env.update_env_with_params(
            width=40,
            height=40,
            num_agents=2,
            max_steps = 250,
            rail_type = 'sparse',
            rail_gen_params = {
                'num_cities': 2,
                'grid_mode': False,
                'max_rails_between_cities': 1,
                'max_rails_in_city' : 2
            },
            seed = self.seed       
        )


    def change_grid_round2(self, env):
        env.update_env_with_params(
            width=40,
            height=40,
            num_agents=3,
            max_steps = 300,
            rail_type = 'sparse',
            rail_gen_params = {
                'num_cities': 3,
                'grid_mode': False,
                'max_rails_between_cities': 1,
                'max_rails_in_city' : 2
            },
            seed = self.seed       
        )
    
    def change_grid_round3(self, env):
        env.update_env_with_params(
            width=50,
            height=50,
            num_agents=4,
            max_steps = 450,
            rail_type = 'sparse',
            rail_gen_params = {
                'num_cities': 4,
                'grid_mode': False,
                'max_rails_between_cities': 1,
                'max_rails_in_city' : 2
            },
            seed = self.seed   
        )

    def change_grid_round4(self, env):
        env.update_env_with_params(
            width=70,
            height=70,
            num_agents=10,
            max_steps = 450,
            rail_type = 'sparse',
            rail_gen_params = {
                'num_cities': 10,
                'grid_mode': False,
                'max_rails_between_cities': 2,
                'max_rails_in_city' : 3
            },
            seed = self.seed       
        )

    def change_grid_round5(self, env):
        env.update_env_with_params(
            width=100,
            height=100,
            num_agents=20,
            max_steps = 500,
            rail_type = 'sparse',
            rail_gen_params = {
                'num_cities': 15,
                'grid_mode': False,
                'max_rails_between_cities': 2,
                'max_rails_in_city' : 4
            },
            seed = self.seed       
        )

