import numpy as np
import time
import copy

from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool
from flatland.envs.observations import TreeObsForRailEnv, LocalObsForRailEnv
from flatland.envs.generators import complex_rail_generator


NUMBER_OF_AGENTS = 10
env = RailEnv(
            width=20,
            height=20,
            rail_generator=complex_rail_generator(
                                    nr_start_goal=10,
                                    nr_extra=1,
                                    min_dist=8,
                                    max_dist=99999,
                                    seed=0),
            number_of_agents=NUMBER_OF_AGENTS,
            obs_builder_object=LocalObsForRailEnv(3,3,0))

env_renderer = RenderTool(env)

def my_controller():
    _action = {}
    for _idx in range(NUMBER_OF_AGENTS):
        _action[_idx] = np.random.randint(0, 5)
    return _action

for step in range(100):
    _action = my_controller()
    obs, all_rewards, done, _ = env.step(_action)

    print("Rewards: {}, [done={}]".format( all_rewards, done))
    env_renderer.render_env(show=True, frames=False, show_observations=True)
    time.sleep(0.3)