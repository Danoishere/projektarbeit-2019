import constant as const
import requests

class Curriculum:
    def __init__(self):
        self.curriculum = {
            0: {
                'level_generator' : lambda env: self.change_grid_round0(env),
                'switch_on_successrate': 0.95
                },
            1: {
                'level_generator' : lambda env: self.change_grid_round1(env),
                'switch_on_successrate': 0.95
                },
            2: {
                'level_generator' : lambda env: self.change_grid_round2(env),
                'switch_on_successrate': 0.95
                },
            3: {
                'level_generator' : lambda env: self.change_grid_round3(env),
                'switch_on_successrate': 0.95
                },
            4: {
                'level_generator' : lambda env: self.change_grid_round4(env),
                'switch_on_successrate': 0.95
                },
            5: {
                'level_generator' : lambda env: self.change_grid_round5(env),
                'switch_on_successrate': 0.95
                }
        }

        self.number_of_levels = len(self.curriculum)
        self.update_curriculum_level()
        
    def should_switch_level(self, successrate):
        switch_on_successrate = self.curriculum[self.current_level]
        return successrate >= switch_on_successrate
            

    def update_env_to_curriculum_level(self, env):
        return self.curriculum[self.current_level]['level_generator'](env)


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

    def change_grid_round1(self, env):
        env.update_env_with_params(
            width=30,
            height=30,
            num_agents=2,
            max_steps = 100,
            rail_type = 'sparse',
            rail_gen_params = {
                'num_cities': 2,
                'grid_mode': False,
                'max_rails_between_cities': 1,
                'max_rails_in_city' : 2
            }
        )


    def change_grid_round2(self, env):
        env.update_env_with_params(
            width=40,
            height=40,
            num_agents=3,
            max_steps = 100,
            rail_type = 'sparse',
            rail_gen_params = {
                'num_cities': 3,
                'grid_mode': False,
                'max_rails_between_cities': 1,
                'max_rails_in_city' : 2
            }
        )
    
    def change_grid_round3(self, env):
        env.update_env_with_params(
            width=50,
            height=50,
            num_agents=4,
            max_steps = 140,
            rail_type = 'sparse',
            rail_gen_params = {
                'num_cities': 4,
                'grid_mode': False,
                'max_rails_between_cities': 1,
                'max_rails_in_city' : 2
            }
        )

    def change_grid_round4(self, env):
        env.update_env_with_params(
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

    def change_grid_round5(self, env):
        env.update_env_with_params(
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

