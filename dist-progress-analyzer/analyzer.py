#!/usr/bin/env python
# coding: utf-8

# This iPython notebook includes an implementation of the [A3C algorithm](https://arxiv.org/pdf/1602.01783.pdf).
# 
# tensorboard --logdir=deliverables/tensorboard
#
#  ##### Enable autocomplete

# import shared directory
import os, sys; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)) + os.sep + 'shared')
import pandas as pd
import multiprocess as mp
import numpy as np
import tensorflow as tf
from ctypes import c_bool

from time import sleep, time
from datetime import datetime
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
    urllib.request.urlretrieve(const.url + '/file/network.pyx', 'deliverables/network.pyx')
    urllib.request.urlretrieve(const.url + '/file/input_params.py', 'deliverables/input_params.py')
    urllib.request.urlretrieve(const.url + '/file/observation.pyx', 'deliverables/observation.pyx')
    urllib.request.urlretrieve(const.url + '/file/curriculum.py', 'deliverables/curriculum.py')

    myCmd = 'python setup_deliverables.py build_ext --inplace'
    os.system(myCmd)
    
    network_mod = __import__("deliverables.network", fromlist=[''])
    network_class = getattr(network_mod, 'AC_Network')

    #Create the local copy of the network and the tensorflow op to copy global paramters to local network
    model = network_class(True,const.url)
    env = RailEnvWrapper(model.get_observation_builder())
    
    num_repeat = 20

    df = pd.DataFrame()

    '''
    env_renderer = RenderTool(env.env, gl="PILSVG",
                          agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                          show_debug=False,
                          screen_height=800,  # Adjust these parameters to fit your resolution
                          screen_width=800)
    '''

    num_agents = 14
    round = 0

    while True:
        num_success = 0
        episode_count = 0
        agents_started = 0
        agents_arrived = 0
        model.update_from_global_model()
        print('Model updated from server.')
        prep_steps = 0

        env.update_env_with_params(
            width=50,
            height=50,
            num_agents=num_agents,
            max_steps = 450,
            rail_type = 'sparse',
            rail_gen_params = {
                'num_cities': 10,
                'grid_mode': False,
                'max_rails_between_cities': 3,
                'max_rails_in_city' : 4
            },
            seed = 0
        )

        for r in range(num_repeat):
            episode_done = False

            # Metrics for tensorboard logging
            episode_reward = 0
            episode_step_count = 0
            
            obs, info = env.reset()
            obs_builder = env.env.obs_builder

            prep_steps = 0
            done = {i:False for i in range(len(env.env.agents))}
            done['__all__'] = False
            all_handles = [i for i in range(len(env.env.agents))]
            no_reward = {i:0 for i in range(len(env.env.agents))}

            agent_pos = {}

            max_steps = env.env._max_episode_steps - 5
            while episode_done == False and episode_step_count < max_steps:
                agents = env.env.agents
                env_actions, nn_actions, v, relevant_obs = model.get_agent_actions(env.env, obs, info, False)

                next_obs, rewards, done, info = env.step(env_actions)

                #env_renderer.render_env(show=True)

                handles = []
                for agent in agents:
                    if agent.position is not None:
                        handles.append((agent.handle, *agent.position, agent.malfunction_data['malfunction'], agent.wait))

                agent_pos_key = tuple(handles)
                if agent_pos_key in agent_pos:
                    agent_pos[agent_pos_key] += 1
                else:
                    agent_pos[agent_pos_key] = 0

                max_pos_repeation = max(agent_pos.values())
                if max_pos_repeation > 10:
                    cancel_episode = True

                prep_steps = 0
                obs_builder.prep_steps = prep_steps
                episode_step_count += 1

                episode_done = done['__all__']
                if episode_done == True:
                    next_obs = obs
            
                obs = next_obs              
                done_last_step = dict(done)         
                

            agents_started += len(agents)
            episode_count += 1
            if episode_done:
                num_success += 1

            num_successful_agents = 0
            for i in range(env.num_agents):
                if done[i]:
                    num_successful_agents += 1
                    episode_reward += 1.0
                    agents_arrived += 1

            current_successrate = agents_arrived/agents_started
            print('Eval. episode', episode_count,'with',episode_step_count,'steps, reward of',episode_reward,', agents arrived',current_successrate)

            df = df.append({
                    'round' : round,
                    'departed' : num_agents,
                    'arrived' : agents_arrived,
                    'reward' : episode_reward,
                    'steps' : episode_step_count,
                    'time' : datetime.now()
                }, ignore_index=True)
            
        df.to_csv('analysis_round_' + str(round) + '.csv')
        successrate = num_success/num_repeat
        round += 1

        print('Analyzation round finished. sucessrate (agents arrived/agents started):', current_successrate)
        

    print ("Looks like we're done")


def is_agent_on_usable_switch(env, position, dir):
        ''' a tile is a switch with more than one possible transitions for the
            given direction. '''

        if position is None:
            return False

        transition = env.env.rail.get_transitions(*position, dir)

        if np.sum(transition) == 1:
            return False
        else:
            return True

def is_agent_on_unusable_switch(env, position, dir):
    ''' a tile is a switch with more than one possible transitions for the
        given direction. '''

    if position is None:
        return False

    possible_transitions = np.sum(env.env.rail.get_transitions(*position, dir))
    #print(env.rail.get_transitions(*position, dir))
    for d in range(4):
        dir_transitions = np.sum(env.env.rail.get_transitions(*position, d))
        if dir_transitions > possible_transitions >= 1:
            #print(env.rail.get_transitions(*position, d))
            return True

    return False

def agent_action_to_env_action(env, agent, agent_action):
    ''' agent actions: left, right, wait
        env actions: 'do nothing, left, forward, right, brake 
    '''
    if agent.position is None:
        # Ready to depart. Wait or go?
        if agent_action == 3:
            return RailEnvActions.MOVE_FORWARD
        else:
            return RailEnvActions.DO_NOTHING

    if is_agent_on_unusable_switch(env, agent.next_pos, agent.direction):
        if agent_action == 3:
            return RailEnvActions.MOVE_FORWARD
        else:
            if agent.speed_data['speed'] > 0:
                return RailEnvActions.STOP_MOVING
            else:
                return RailEnvActions.DO_NOTHING

    if agent_action == 3:
        return RailEnvActions.DO_NOTHING

    if agent_action == 2:
        agent.wait = 5
        if agent.speed_data['speed'] > 0:
            return RailEnvActions.STOP_MOVING
        else:
            return RailEnvActions.DO_NOTHING

    dir = agent.direction
    transition = env.env.rail.get_transitions(*agent.position, agent.direction)

    can_go_left = False
    can_go_forward = False
    can_go_right = False

    if transition[(3 + dir) % 4] == 1:
        can_go_left = True
    if transition[(0 + dir) % 4] == 1:
        can_go_forward = True
    if transition[(1 + dir) % 4] == 1:
        can_go_right = True

    # print('Can go left:', can_go_left)
    # print('Can go forward:', can_go_forward)
    # print('Can go right:', can_go_right)
    
    if agent_action == 0 and can_go_left:
        return RailEnvActions.MOVE_LEFT
    if agent_action == 1 and can_go_right:
        return RailEnvActions.MOVE_RIGHT

    return RailEnvActions.MOVE_FORWARD


def next_pos(env, position, direction):
    if position is None:
        return None

    transition = env.env.rail.get_transitions(*position, direction)
    if np.sum(transition) > 1:
        None

    posy = position[0] - transition[0]  + transition[2]
    posx = position[1] + transition[1] - transition[3]

    return [posy, posx]

if __name__ == "__main__":
    start_train(False)
