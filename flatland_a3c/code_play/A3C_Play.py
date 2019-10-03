#!/usr/bin/env python
# coding: utf-8

# This iPython notebook includes an implementation of the [A3C algorithm](https://arxiv.org/pdf/1602.01783.pdf).
# 
# `tensorboard --logdir=worker_0:./train_0',worker_1:./train_1,worker_2:./train_2,worker_3:./train_3`
#
#  ##### Enable autocomplete

import sys
sys.path.append('..')

import json
import threading
from multiprocessing import cpu_count
from multiprocessing.queues import JoinableQueue
from random import choice
from time import sleep, time

import multiprocess
import numpy as np
import scipy.signal

from flatland.utils.rendertools import RenderTool
from multiprocess import JoinableQueue, Manager, Process, Queue
from tensorflow.keras import optimizers

from rail_env_wrapper import RailEnvWrapper
from code_input.observation import CombinedObservation

class Consumer(multiprocess.Process):
    def __init__(self, name, task_queue, result_queue, model_path, model_name):
        print('Created')
        multiprocess.Process.__init__(self)
        self.name = name
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.model_name = model_name
        self.model_path = model_path


    def run(self):
        from tensorflow.keras import layers, models, optimizers
        import code_play.MCRunner as mc
        import numpy as np

        self.model = models.load_model(self.model_path + '/' + self.model_name)
        self.np = np

        print('Model loaded in ' + self.name)
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                print('{}: Exiting'.format(proc_name))
                self.task_queue.task_done()
                break
            print('Run accepted on ' + self.name)
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
        num_agents = 16
                    
        #The Below code is related to setting up the Flatland environment
        self.env = RailEnvWrapper(
            width=40,
            height=40,
            num_agents=num_agents
        )

        self.env.step_penalty = -2
        self.env.global_reward = 10

        self.renderer = RenderTool(self.env.env)
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
                    obs, rewards, done = self.env.step(actions)
                    steps += 1
                    episode_done = done['__all__']
                    episode_steps_count += 1

                    # Stuck detection
                    agents_state = tuple([i.position for i in self.env.env.agents])
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


max_episode_length = 80
mc_sim_num = 1
mc_sim_steps = 8
mc_max_rollout = 6

# Observation sizes
map_state_size = (11,11,9)
grid_state_size = (11,11,16)
vector_state_size = 5

# Action size
a_size = 5 # Agent can move Left, Right, or Fire

load_model = True
model_path = './model'
model_name =  'model-6000.h5'
processes = []


def start_play():
    m = Manager()
    tasks = m.JoinableQueue()
    results = m.Queue()
    for ncpu in range(cpu_count()):
        consumer = Consumer('runner_' + str(ncpu), tasks, results, model_path, model_name)
        processes.append(consumer)
        consumer.start()

    multiprocess.freeze_support()
    model = models.load_model(model_path + '/' + model_name)

    player = Player(model, tasks, results)
    player.play(max_episode_length)
    print ("Looks like we're done")
