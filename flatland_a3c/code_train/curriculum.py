class CurriculumManager:
    def __init__(self, coordinator, authorized_worker):
        self.coordinator = coordinator
        self.current_level = 0
        self.next_level = 1
        self.stop_training = False
        self.authorized_worker = authorized_worker
        self.curriculum = {
            0: {
                'next_after_successrate' : 0.95,
                'level_generator' : lambda env: self.change_grid_round0(env)
            },
            1: {
                'next_after_successrate' : 0.95,
                'level_generator' : lambda env: self.change_grid_round1(env)
            },
            2: {
                'next_after_successrate' : 0.95,
                'level_generator' : lambda env: self.change_grid_round2(env)
            },
            3: {
                'next_after_successrate' : 0.95,
                'level_generator' : lambda env: self.change_grid_round3(env)
            },
            4: {
                'next_after_successrate' : 0.95,
                'level_generator' : lambda env: self.change_grid_round4(env)
            },
            5: {
                'next_after_successrate' : 1,
                'level_generator' : lambda env: self.change_grid_round5(env)
            }
        }

        self.number_of_levels = len(self.curriculum)
        

    def update_env_to_curriculum_level(self, env):
        return self.curriculum[self.current_level]['level_generator'](env)


    def report_success_rate(self, success_rate, worker_name):
        if self.authorized_worker != worker_name:
            return

        if success_rate >= self.curriculum[self.current_level]['next_after_successrate']:
            self.next_level = self.current_level + 1
            self.coordinator.request_stop()

    def switch_to_next_level(self):
        if self.next_level == self.number_of_levels:
            self.stop_training = True
        else:
            self.current_level = self.next_level

    def change_grid_round0(self, env):
        env.update_env_with_params(
            width=8,
            height=8,
            num_agents=1,
            max_steps = 40,
            rail_type = 'complex',
            rail_gen_params = {
                'nr_start_goal': 3,
                'nr_extra': 3,
                'min_dist': 8,
                'max_dist' : 99999
            }
        )

    def change_grid_round1(self, env):
        env.update_env_with_params(
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
            }
        )


    def change_grid_round2(self, env):
        env.update_env_with_params(
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
            }
        )
    
    def change_grid_round3(self, env):
        env.update_env_with_params(
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
            }
        )

    def change_grid_round4(self, env):
        env.update_env_with_params(
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
                'num_neighb': 3, # Number of connections to other cities
                'grid_mode': True,
                'enhance_intersection':True
            }
        )

    def change_grid_round5(self, env):
        env.update_env_with_params(
            width=60,
            height=60,
            num_agents=6,
            max_steps = 40,
            rail_type = 'sparse',
            rail_gen_params = {
                'num_cities': 12,  # Number of cities in map
                'num_intersections': 15,  # Number of interesections in map
                'num_trainstations': 20,  # Number of possible start/targets on map
                'min_node_dist': 10,  # Minimal distance of nodes
                'node_radius': 3, # Proximity of stations to city center
                'num_neighb': 5, # Number of connections to other cities
                'grid_mode': False,
                'enhance_intersection':False
            }
        )
