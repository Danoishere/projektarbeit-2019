import sys
sys.path.append("flatland")

from flatland.evaluators.client import FlatlandRemoteClient
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.generators import random_rail_generator, complex_rail_generator
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.transition_map import GridTransitionMap

import numpy as np


import argparse
from linker import FlatLink, FlatConvert
from fake import fakeEnv

import copy
import numpy as np
import configparser

from observation import RawObservation


#from flatland.utils.rendertools import RenderTool
import time
import pickle



rt = None
config = configparser.ConfigParser()
config.read('flatland_a3c.ini')

flatconfig = config['ENVIRONMENT']

size_x=int(flatconfig['size_x'])
size_y=int(flatconfig['size_y'])
fov_size_x = int(flatconfig['fov_size_x'])
fov_size_y = int(flatconfig['fov_size_y'])
num_agents = int(flatconfig['num_agents'])
num_extra = int(flatconfig['num_extra'])
min_dist_ = int(flatconfig['min_dist_'])

gpu_=-1

if config['GPU_SUPPORT'].getboolean('UseGpu'):
    gpu_ = int(config['GPU_SUPPORT']['GpuID'])


link_ = FlatConvert(fov_size_x,fov_size_y)
link_.gpu_ = gpu_

hyperparams = config['HYPERPARAMS']

link_.play_epochs = int(hyperparams['play_epochs'])
link_.n_step_return_ = int(hyperparams['n_step_return_'])
link_.play_game_number = int(hyperparams['play_game_number'])
link_.number_of_levels = int(flatconfig['number_of_levels'])
link_.replay_buffer_size = int(hyperparams['replay_buffer_size'])
link_.num_epochs = int(hyperparams['num_epochs'])
link_.batch_size = int(hyperparams['batch_size'])
link_.buffer_size = int(hyperparams['buffer_size'])
link_.time_stack = int(hyperparams['time_stack'])
link_.max_iter = int(hyperparams['max_iter'])
link_.gamma_ = float(hyperparams['gamma_'])
link_.epsilon_start = float(hyperparams['epsilon_start'])
link_.epsilon_end = float(hyperparams['epsilon_end'])
link_.buffer_threshold = float(hyperparams['buffer_threshold'])

link_.init_model()
link_.load_cpu('models/d2/model_a3c_i.pt')

#####################################################################
# Instantiate a Remote Client
#####################################################################
remote_client = FlatlandRemoteClient()

#####################################################################
# Define your custom controller
#
# which can take an observation, and the number of agents and 
# compute the necessary action for this step for all (or even some)
# of the agents
#####################################################################
def my_controller(obs, number_of_agents):
    
    return link_.step(obs)

#####################################################################
# Instantiate your custom Observation Builder
# 
# You can build your own Observation Builder by following 
# the example here : 
# https://gitlab.aicrowd.com/flatland/flatland/blob/master/flatland/envs/observations.py#L14
#####################################################################
my_observation_builder = RawObservation([fov_size_x,fov_size_y])

#####################################################################
# Main evaluation loop
#
# This iterates over an arbitrary number of env evaluations
#####################################################################
evaluation_number = 0
while True:

    evaluation_number += 1
    # Switch to a new evaluation environemnt
    # 
    # a remote_client.env_create is similar to instantiating a 
    # RailEnv and then doing a env.reset()
    # hence it returns the first observation from the 
    # env.reset()
    # 
    # You can also pass your custom observation_builder object
    # to allow you to have as much control as you wish 
    # over the observation of your choice.
    observation = remote_client.env_create(
                    obs_builder_object=my_observation_builder
                )
    if not observation:
        #
        # If the remote_client returns False on a `env_create` call,
        # then it basically means that your agent has already been 
        # evaluated on all the required evaluation environments,
        # and hence its safe to break out of the main evaluation loop
        break
    
    print("Evaluation Number : {}".format(evaluation_number))

    #####################################################################
    # Access to a local copy of the environment
    # 
    #####################################################################
    # Note: You can access a local copy of the environment 
    # by using : 
    #       remote_client.env 
    # 
    # But please ensure to not make any changes (or perform any action) on 
    # the local copy of the env, as then it will diverge from 
    # the state of the remote copy of the env, and the observations and 
    # rewards, etc will behave unexpectedly
    # 
    # You can however probe the local_env instance to get any information
    # you need from the environment. It is a valid RailEnv instance.
    local_env = remote_client.env
    number_of_agents = len(local_env.agents)

    # Now we enter into another infinite loop where we 
    # compute the actions for all the individual steps in this episode
    # until the episode is `done`
    # 
    # An episode is considered done when either all the agents have 
    # reached their target destination
    # or when the number of time steps has exceed max_time_steps, which 
    # is defined by : 
    #
    # max_time_steps = int(1.5 * (env.width + env.height))
    #
    iter_=0
    while iter_<100:
        iter_+=1
        #####################################################################
        # Evaluation of a single episode
        #
        #####################################################################
        # Compute the action for this step by using the previously 
        # defined controlle
        action = my_controller(observation, number_of_agents)

        # Perform the chosen action on the environment.
        # The action gets applied to both the local and the remote copy 
        # of the environment instance, and the observation is what is 
        # returned by the local copy of the env, and the rewards, and done and info
        # are returned by the remote copy of the env
        observation, all_rewards, done, info = remote_client.env_step(action)
        if done['__all__']:
            print("Reward : ", sum(list(all_rewards.values())))
            #
            # When done['__all__'] == True, then the evaluation of this 
            # particular Env instantiation is complete, and we can break out 
            # of this loop, and move onto the next Env evaluation
            break
    print("Episode ",number_of_agents)
print("Evaluation of all environments complete...")
########################################################################
# Submit your Results
# 
# Please do not forget to include this call, as this triggers the 
# final computation of the score statistics, video generation, etc
# and is necesaary to have your submission marked as successfully evaluated
########################################################################
#print(remote_client.submit())