#!/usr/bin/env python
# coding: utf-8

# This iPython notebook includes an implementation of the [A3C algorithm](https://arxiv.org/pdf/1602.01783.pdf).
# 
# tensorboard --logdir=deliverables/tensorboard
#
#  ##### Enable autocomplete

# import shared directory
import os, sys; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)) + os.sep + 'shared')

import multiprocess as mp
import numpy as np
import tensorflow as tf
from ctypes import c_bool

from time import sleep, time
from rail_env_wrapper import RailEnvWrapper

import constant as const
import urllib
import requests

mp.set_start_method('spawn', True)

def get_curriculum_lvl():
    data = requests.get(url=const.url + '/').json()
    return data['curriculum_lvl']


def start_train(resume):
    
    urllib.request.urlretrieve(const.url + '/network_file', 'deliverables/network.py')
    urllib.request.urlretrieve(const.url + '/config_file', 'deliverables/input_params.py')
    urllib.request.urlretrieve(const.url + '/observation_file', 'deliverables/observation.py')
    urllib.request.urlretrieve(const.url + '/curriculum_file', 'deliverables/curriculum.py')

    curriculum_mod = __import__("deliverables.curriculum", fromlist=[''])
    curriculum_class =  getattr(curriculum_mod, 'Curriculum')
    curriculum = curriculum_class()

    # Initial receiving of curriculum-level. Maybe the server is running already and
    # progress has been made
    curriculum.update_curriculum_level()
    
    network_mod = __import__("deliverables.network", fromlist=[''])
    network_class = getattr(network_mod, 'AC_Network')
    obs_helper = __import__("deliverables.observation", fromlist=[''])
    params = __import__("deliverables.input_params", fromlist=[''])

    #Create the local copy of the network and the tensorflow op to copy global paramters to local network
    model = network_class(True,const.url)
    env = RailEnvWrapper(model.get_observation_builder())

    curriculum.update_env_to_curriculum_level(env)
    num_repeat = 100

    while True:

        num_success = 0
        episode_count = 0
        model.update_from_global_model()
        print('Model updated from server.')

        for r in range(num_repeat):
            episode_done = False
            # Buffer for obs, action, next obs, reward
            episode_buffer = [[] for i in range(env.num_agents)]

            # A way to remember if the agent was already done during the last episode
            done_last_step = {i:False for i in range(env.num_agents)}

            # Metrics for tensorboard logging
            episode_reward = 0
            episode_step_count = 0
            
            obs, info = env.reset()

            while episode_done == False and episode_step_count < env.max_steps:
                actions = model.get_best_actions(obs)
                next_obs, rewards, done, info = env.step(actions)

                episode_done = done['__all__']
                if episode_done == True:
                    next_obs = obs

                for i in range(env.num_agents):
                    agent_obs = obs[i] 
                    agent_action = actions[i]
                    agent_reward = rewards[i]
                    agent_next_obs =  next_obs[i]

                    if not done_last_step[i]:
                        episode_buffer[i].append([
                            agent_obs,
                            agent_action,
                            agent_reward,
                            agent_next_obs,
                            episode_done,
                            0])
                        
                        episode_reward += agent_reward
                
                obs = next_obs               
                episode_step_count += 1
                done_last_step = dict(done)


            episode_count += 1
            if episode_done:
                num_success += 1

            print('Eval. episode', episode_count,'with',episode_step_count,'steps, reward of',episode_reward,', curriculum level', curriculum.current_level)

        successrate = num_success/num_repeat
        print('Evaluation round finished. sucessrate:', successrate)
        if curriculum.should_switch_level(successrate):
            print('Curriculum level change triggered.')
            # Send new level & receive confirmation
            curriculum.increase_curriculum_level()
            # Update local env to use new level
            curriculum.update_env_to_curriculum_level(env)
        

    print ("Looks like we're done")

if __name__ == "__main__":
    start_train(False)
