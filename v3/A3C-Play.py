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


import json
import threading
from multiprocessing import cpu_count
from multiprocessing.queues import JoinableQueue
from random import choice
from time import sleep, time

import matplotlib.pyplot as plt
import multiprocess
import numpy as np
import scipy.signal
from flatland.core.grid.grid4_astar import a_star
from flatland.envs.observations import (GlobalObsForRailEnv,
                                        LocalObsForRailEnv, TreeObsForRailEnv)
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator
from flatland.utils.rendertools import RenderTool
from multiprocess import JoinableQueue, Manager, Process, Queue
from tensorflow.keras import layers, models, optimizers

from helper import *
from observation import RawObservation

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
    import numpy as np
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

class Consumer(multiprocess.Process):
    def __init__(self, task_queue, result_queue, model_path):
        print('Created')
        multiprocess.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.model_path = model_path

        

    def run(self):
        from tensorflow.keras import layers, models, optimizers
        import MCRunner as mc
        import numpy as np

        self.model = models.load_model(self.model_path + '/' +'model-400.h5')
        self.np = np

        print('Waitin')
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                print('{}: Exiting'.format(proc_name))
                self.task_queue.task_done()
                break

            obs = next_task[0]
            env = next_task[1]
            max_attempt_length = next_task[2]

            result = mc.run_attempt(obs, self.model, env, max_attempt_length)

            self.task_queue.task_done()
            self.result_queue.put(result)



class Player():
    def __init__(self, model, tasks, results):
        self.name = "player"    
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_success = []
        self.episode_mean_values = []

        self.tasks = tasks
        self.results = results

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.model = model
        num_agents = 3
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
                 
    
    def play(self,max_episode_length):
        total_attempts = 0
        episode_count = 0
          
        while True:
            episode_steps_count = 0
            episode_reward = 0
            episode_done = False

            self.env.step_penalty = -2
            obs = self.env.reset()
            obs = reshape_obs(obs)

            while obs[0].shape[0] == 0:
                obs = self.env.reset()
                obs = reshape_obs(obs)
                print('Regenerate environment - no suitable solution found')

            self.renderer.set_new_rail()

            pos = {}
            while not episode_done and episode_steps_count < max_episode_length:
                rewards = []
                action_lists = []
                for i in range(mc_sim_num):
                    self.tasks.put((obs, self.env, mc_sim_steps))
                
                attempt_results = []
                while len(attempt_results) != mc_sim_num:
                    attempt_results.append(self.results.get())

                for r in attempt_results:
                    action_lists.append(r[2])
                    rewards.append(r[1])

                best_action_idx = np.argmax(rewards)
                best_action = action_lists[best_action_idx]

                episode_done = False
                steps = 0
                
                for actions in best_action[:mc_max_rollout]:
                    sleep(0.2)
                    self.renderer.render_env(show=True, frames=False, show_observations=False)
                    obs, rewards, done, _ = self.env.step(actions)
                    obs = reshape_obs(obs)
                    steps += 1
                    episode_done = done['__all__']
                    episode_steps_count += 1

                    # Stuck detection
                    agents_state = tuple([i.position for i in self.env.agents])
                    if agents_state in pos.keys():
                        pos[agents_state] += 1
                    else:
                        pos[agents_state] = 1
        
                    is_stuck = False
                    for state in pos:
                        if pos[state] > 8:
                            is_stuck = True
                            episode_steps_count = max_episode_length
                            print('Stuck. Abort simulation')
                            break

                    if is_stuck:
                        break

                if is_stuck:
                    break

            episode_count += 1
            print('Episode', episode_count,', finished=', episode_done, 'of',self.name,'with', episode_steps_count ,'steps, reward of',episode_reward)
    
# In[23]:

max_episode_length = 80
mc_sim_num = 8
mc_sim_steps = 25
mc_max_rollout = 6

# Observation sizes
map_state_size = (11,11,9)
grid_state_size = (11,11,16)
vector_state_size = 5

# Action size
a_size = 5 # Agent can move Left, Right, or Fire

load_model = True
model_path = './model'
processes = []



if __name__ == "__main__":
    m = Manager()
    tasks = m.JoinableQueue()
    results = m.Queue()
    for ncpu in range(cpu_count()):
        consumer = Consumer(tasks, results, model_path)
        processes.append(consumer)
        consumer.start()

    multiprocess.freeze_support()
    model = models.load_model(model_path + '/' +'model-400.h5')

    player = Player(model, tasks, results)
    player.play(max_episode_length)
    print ("Looks like we're done")
