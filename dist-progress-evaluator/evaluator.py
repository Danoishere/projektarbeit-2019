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
from flatland.envs.rail_env import RailEnvActions, RailAgentStatus

import constant as const
import urllib
import requests

#from flatland.utils.rendertools import RenderTool, AgentRenderVariant

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

    #Create the local copy of the network and the tensorflow op to copy global paramters to local network
    model = network_class(True,const.url)
    env = RailEnvWrapper(model.get_observation_builder())

    curriculum.update_env_to_curriculum_level(env)
    curriculum.seed = 12345
    num_repeat = 100

    '''
    env_renderer = RenderTool(env.env, gl="PILSVG",
                          agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                          show_debug=False,
                          screen_height=800,  # Adjust these parameters to fit your resolution
                          screen_width=800)
    '''
    

    while True:

        num_success = 0
        episode_count = 0
        model.update_from_global_model()
        print('Model updated from server.')
        prep_steps = 0

        for r in range(num_repeat):
            episode_done = False

            # Metrics for tensorboard logging
            episode_reward = 0
            episode_step_count = 0
            
            obs, info = env.reset()
            # env_renderer.set_new_rail()
            obs_builder = env.env.obs_builder

            prep_steps = 0
            done = {i:False for i in range(len(env.env.agents))}
            done['__all__'] = False
            all_handles = [i for i in range(len(env.env.agents))]
            no_reward = {i:0 for i in range(len(env.env.agents))}

            while episode_done == False and episode_step_count < env.max_steps:
                obs_dict = {}
                for handle in range(len(env.env.agents)):
                    if info['status'][handle] == RailAgentStatus.READY_TO_DEPART or (
                        info['action_required'][handle] and info['malfunction'][handle] == 0):
                        obs_dict[handle] = obs[handle] 

                actions,_ = model.get_best_actions_and_values(obs_dict, env.env)

                if prep_steps == 1:
                    next_obs, rewards, done, info = env.step(actions)
                    for agent in env.env.agents:
                        agent.last_action = 0

                    prep_steps = 0
                else:
                    prep_steps += 1
                    next_obs = env.env.obs_builder.get_many(all_handles)
                    rewards = dict(no_reward)

                #env_renderer.render_env(show=True)

                episode_done = done['__all__']
                if episode_done == True:
                    next_obs = obs

                for i in range(env.num_agents):
                    episode_reward += rewards[i]
                
                obs = next_obs               
                episode_step_count += 1

            episode_count += 1
            if episode_done:
                num_success += 1

            for i in range(env.num_agents):
                if done[i]:
                    # If agents could finish the level, 
                    # set final reward for all agents
                    episode_buffer[i][-1][2] += 1.0
                    episode_reward += 1.0

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
