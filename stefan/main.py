"""
Author: S.Huschauer
Date: 31.08.2019
Email: huschste@students.zhaw.ch
Associated with ZHAW datalab
This code is not exhaustive and "is as it is"!
"""
import argparse
from linker import FlatLink, FlatConvert
from fake import fakeEnv
import sys
import copy
import numpy as np
import configparser
sys.path.append("flatland")
from observation import RawObservation

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import random_rail_generator, complex_rail_generator
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.transition_map import GridTransitionMap
from flatland.utils.rendertools import RenderTool
import time
import pickle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #define possible choices
    parser.add_argument('action', choices=['generate','genasync','play'])



    args = parser.parse_args()
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

    if args.action == 'generate':
        #sequential training
        env = RailEnv(size_x,size_y,rail_generator=complex_rail_generator(num_agents,nr_extra=num_extra, min_dist= min_dist_,seed=125),number_of_agents=num_agents,obs_builder_object=RawObservation([fov_size_x,fov_size_y]))
        # instatiate renderer if needed
        rt = None #RenderTool(env,gl="PILSVG")
        link_.perform_training(env,rt)
       
    elif args.action == 'genasync':
        # asyncronous training
        # attention uses python multiprocessing
        link_.asynchronous_training(size_x,size_y,num_agents)

    elif args.action == 'play':
        # play game with trained data
        env = RailEnv(size_x,size_y,rail_generator=complex_rail_generator(num_agents,nr_extra=num_extra, min_dist=min_dist_,seed=125),number_of_agents=num_agents,obs_builder_object=RawObservation([fov_size_x,fov_size_y]))
        rt = RenderTool(env,gl="PILSVG")
        link_.play(env,"model_a3c_i.pt",rt)

