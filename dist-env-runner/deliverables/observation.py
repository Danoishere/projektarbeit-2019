import numpy as np
import deliverables.input_params as params
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

"""
Collection of environment-specific ObservationBuilder.
"""
import collections
from typing import Optional, List, Dict, Tuple

import numpy as np
from time import time

from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.agent_utils import RailAgentStatus, EnvAgent
from flatland.utils.ordered_set import OrderedSet
from flatland.envs.rail_env import RailEnvActions


class RailObsBuilder(ObservationBuilder):
    def __init__(self):
        super().__init__()
        self.actor_rec_state = {}
        self.critic_rec_state = {}
        self.prep_step = 0
        self.comm = np.zeros(5)

    def reset(self):
        self.actor_rec_state = {}
        self.critic_rec_state = {}
        self.prep_step = 0
        self.comm = np.zeros(5)

    def get_many(self, handles=None):
        obs_dict = {}
        for agent in self.env.agents:
            agent_obs = self.comm

            if agent.handle not in  self.actor_rec_state:
                self.actor_rec_state[agent.handle] = np.zeros((2,params.recurrent_size))

            if agent.handle not in  self.critic_rec_state:
                self.critic_rec_state[agent.handle] = np.zeros((2,params.recurrent_size))
            
            rec_actor = self.actor_rec_state[agent.handle]
            rec_critic = self.critic_rec_state[agent.handle]
            obs_dict[agent.handle] = (agent_obs, rec_actor, rec_critic)

        return obs_dict

    def buffer_to_obs_lists(self, episode_buffer):
        vec_obs = np.asarray([row[0][0] for row in episode_buffer])
        a_rec_obs = np.asarray([row[0][1] for row in episode_buffer])
        c_rec_obs = np.asarray([row[0][2] for row in episode_buffer])

        return vec_obs, a_rec_obs, c_rec_obs