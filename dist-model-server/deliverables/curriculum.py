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
                'switch_on_successrate': 0.70
                },
            1: {
                'level_generator' : lambda env: self.change_grid_round1(env),
                'switch_on_successrate': 0.70
                },
            2: {
                'level_generator' : lambda env: self.change_grid_round2(env),
                'switch_on_successrate': 0.70
                },
            3: {
                'level_generator' : lambda env: self.change_grid_round3(env),
                'switch_on_successrate': 0.70
                },
            4: {
                'level_generator' : lambda env: self.change_grid_round4(env),
                'switch_on_successrate': 0.65
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
            width=50,
            height=50,
            num_agents=14,
            max_steps = 1000,
            rail_type = 'sparse',
            rail_gen_params = {
                'num_cities': 10,
                'grid_mode': False,
                'max_rails_between_cities': 2,
                'max_rails_in_city' : 3
            },
            seed = self.seed        
        )

    def change_grid_round1(self, env):
        env.update_env_with_params(
            width=60,
            height=60,
            num_agents=20,
            max_steps = 1000,
            rail_type = 'sparse',
            rail_gen_params = {
                'num_cities': 13,
                'grid_mode': True,
                'max_rails_between_cities': 2,
                'max_rails_in_city' : 3
            },
            seed = self.seed       
        )


    def change_grid_round2(self, env):
        env.update_env_with_params(
            width=70,
            height=70,
            num_agents=50,
            max_steps = 1000,
            rail_type = 'sparse',
            rail_gen_params = {
                'num_cities': 18,
                'grid_mode': True,
                'max_rails_between_cities': 2,
                'max_rails_in_city' : 3
            },
            seed = self.seed       
        )
    
    def change_grid_round3(self, env):
        env.update_env_with_params(
            width=50,
            height=50,
            num_agents=50,
            max_steps = 450,
            rail_type = 'sparse',
            rail_gen_params = {
                'num_cities': 5,
                'grid_mode': True,
                'max_rails_between_cities': 2,
                'max_rails_in_city' : 3
            },
            seed = self.seed   
        )

    def change_grid_round4(self, env):
        env.update_env_with_params(
            width=70,
            height=70,
            num_agents=100,
            max_steps = 600,
            rail_type = 'sparse',
            rail_gen_params = {
                'num_cities': 20,
                'grid_mode': True,
                'max_rails_between_cities': 2,
                'max_rails_in_city' : 4
            },
            seed = self.seed       
        )

    def change_grid_round5(self, env):
        env.update_env_with_params(
            width=100,
            height=100,
            num_agents=200,
            max_steps = 700,
            rail_type = 'sparse',
            rail_gen_params = {
                'num_cities': 25,
                'grid_mode': True,
                'max_rails_between_cities': 3,
                'max_rails_in_city' : 4
            },
            seed = self.seed       
        )

