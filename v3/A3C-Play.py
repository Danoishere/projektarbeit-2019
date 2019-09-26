#!/usr/bin/env python
# coding: utf-8

# This iPython notebook includes an implementation of the [A3C algorithm](https://arxiv.org/pdf/1602.01783.pdf).
# 
# `tensorboard --logdir=worker_0:'./train_0',worker_1:'./train_1',worker_2:'./train_2',worker_3:'./train_3'`
#
#  ##### Enable autocomplete

# In[17]:

#get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')

# In[18]:


import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import models
#get_ipython().run_line_magic('matplotlib', 'inline')
from helper import *
from copy import deepcopy

from random import choice
from time import sleep
from time import time
import json


# In[19]:

from observation import RawObservation

from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool
from flatland.envs.observations import TreeObsForRailEnv, LocalObsForRailEnv, GlobalObsForRailEnv
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator
from flatland.core.grid.grid4_astar import a_star

# ### Helper Functions

# In[20]:

def modify_reward(env, rewards, done, done_last_step, num_of_done_agents, shortest_dist):
    for i in range(env.num_agents):
        if not done_last_step[i] and done[i]:
            num_of_done_agents += 1
            # Hand out some reward to all the agents
            for j in range(env.num_agents):
                rewards[j] += 5  

            # Give some reward to our agent
            rewards[i] += 2**num_of_done_agents * 5

    
    for i in range(env.num_agents):
        agent = env.agents[i]
        path_to_target = agent.path_to_target
        current_path_length = len(path_to_target)
        shortest_path_length = shortest_dist[i]

        # Adding reward for getting closer to goal
        if current_path_length < shortest_path_length:
            rewards[i] +=1
            shortest_dist[i] = current_path_length

        # Subtract reward for getting further away
        if current_path_length > shortest_path_length:
            rewards[i] -= 1
    
    return num_of_done_agents

def reshape_obs(agent_observations):
    map_obs = []
    vec_obs = []
    grid_obs = []
    num_agents = len(agent_observations)

    for i in range(num_agents):
        agent_obs = agent_observations[i]
        map_obs.append(agent_obs[0])
        grid_obs.append(agent_obs[1])
        vec_obs.append(agent_obs[2])
        
    map_obs = np.asarray(map_obs).astype(np.float32)
    map_obs = np.reshape(map_obs,(num_agents, map_state_size[0],map_state_size[1],map_state_size[2]))

    grid_obs = np.asarray(grid_obs).astype(np.float32)
    grid_obs = np.reshape(grid_obs,(num_agents, grid_state_size[0],grid_state_size[1],grid_state_size[2],1))

    vec_obs = np.asarray(vec_obs).astype(np.float32)
    return [map_obs, grid_obs, vec_obs]

# ### Actor-Critic Network


# In[22]:


class Player():
    def __init__(self, model):
        self.name = "player"    
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_success = []
        self.episode_mean_values = []

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.model = model
        num_agents = 2
        rail_gen = complex_rail_generator(
            nr_start_goal=3,
            nr_extra=3,
            min_dist=10,
            seed=random.randint(0,100000)
        )
                    
        #The Below code is related to setting up the Flatland environment
        self.env = RailEnv(
                width=8,
                height=8,
                rail_generator = rail_gen,
                schedule_generator =complex_schedule_generator(),
                number_of_agents=num_agents,
                obs_builder_object=RawObservation([11,11]))

        self.env.step_penalty = -2
        self.env.global_reward = 10
        self.env.num_agents = num_agents

        self.renderer = RenderTool(self.env)
        self.actions = [0,1,2,3,4]


    def copy_env(self, env, num=1):
        #env_str = json.dumps(env)
        if num == 1:
            return deepcopy(env) # json.loads(env_str)   
        else:
            return [ deepcopy(env) for x in range(num)]
                 
    
    def play(self,max_episode_length):
        total_attempts = 0
        episode_count = 0
          
        while True:
            episode_reward = 0
            episode_step_count = 0
            
            
            self.env.step_penalty = -2

            obs = self.env.reset()
            obs = reshape_obs(obs)

            while obs[0].shape[0] == 0:
                obs = self.env.reset()
                obs = reshape_obs(obs)
                print('Regenerate environment - no suitable solution found')

            self.renderer.set_new_rail()
            episode_done = False
            
            
            
            env_copy = self.copy_env(self.env)
            attempt = 0
            
            best_action_list = []
            best_attempt_reward = -1000000
            last_attempt_step_count = max_episode_length
            found_solution = False

            while attempt < 20:
                action_list = []
                attempt_step_count = 0
                attempt_reward = 0
                num_of_done_agents = 0
                episode_done = False
                
                done_last_step = {}                
                dist = {}

                for i in range(self.env.num_agents):
                    done_last_step[i] = False
                    dist[i] = 100

                while not episode_done and attempt_step_count < last_attempt_step_count:
                    #Take an action using probabilities from policy network output.
                    predcition = self.model.predict([obs[0],obs[1],obs[2]])
                    actions = {}
                    for i in range(env_copy.num_agents):
                        a_dist = predcition[0][i]
                        a = np.random.choice([0,1,2,3,4], p = a_dist)
                        actions[i] = a
                    
                    action_list.append(actions)
                    next_obs, rewards, done, _ = env_copy.step(actions)
                    next_obs = reshape_obs(next_obs)
                    num_of_done_agents = modify_reward(env_copy, rewards, done, done_last_step, num_of_done_agents, dist)

                    for i in range(self.env.num_agents):
                        if not done_last_step[i]:
                            attempt_reward += rewards[i]

                    # Is episode finished?
                    episode_done = done['__all__']

                    total_attempts += 1
                    attempt_step_count += 1
                    done_last_step = done

                attempt += 1   

                if attempt_reward > best_attempt_reward:
                    found_solution = True
                    best_action_list = action_list
                    last_attempt_step_count = attempt_step_count
                    print('Best attempt step count:', attempt_step_count)

                

                env_copy = self.copy_env(self.env)
                print('Attempt', attempt,'of',self.name,'with',attempt_step_count,'steps, reward of', attempt_reward)
            
            if found_solution:
                # We found working solution. Now replay!
                episode_done = False
                steps = 0
                while not episode_done and steps < max_episode_length:
                    actions = {}
                    if len(best_action_list) > 0:
                        actions = best_action_list.pop(0)
                    else:
                        predcition = self.model.predict([obs[0],obs[1],obs[2]])
                        for i in range(env_copy.num_agents):
                            a_dist = predcition[0][i]
                            a = np.random.choice([0,1,2,3,4], p = a_dist)
                            actions[i] = a

                    self.renderer.render_env(show=True, frames=False, show_observations=True)
                    obs, rewards, done, _ = self.env.step(actions)
                    obs = reshape_obs(obs)
                    num_of_done_agents = modify_reward(self.env, rewards, done, done_last_step, num_of_done_agents, dist)
                    steps += 1
                    # Is episode finished?
                    episode_done = done['__all__']


            episode_count += 1
            print('Episode', episode_count,', finished=', episode_done, 'of',self.name,'with',episode_step_count,'steps, reward of',episode_reward)
# In[23]:

max_episode_length = 40
gamma = 0.98 # discount rate for advantage estimation and reward discounting

map_state_size = (11,11,9) #  Observations are 21*21 with five channels
grid_state_size = (11,11,16)
vector_state_size = 5

a_size = 5 # Agent can move Left, Right, or Fire
load_model = True
model_path = './model'

# In[24]:

model = models.load_model(model_path + '/' +'model-50.h5')
player = Player(model)
player.play(max_episode_length)
print ("Looks like we're done")




