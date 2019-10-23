import threading
import multiprocess as mp
import numpy as np
import tensorflow as tf
from ctypes import c_bool


import scipy.signal
from tensorflow.keras.optimizers import RMSprop

from datetime import datetime
from random import choice,uniform
from time import sleep
from time import time

import os
cwd = os.getcwd()

from rail_env_wrapper import RailEnvWrapper

import constant as const

class KeyboardInterruptError(Exception): pass

def create_worker(name, should_stop):
    worker = Worker(name, should_stop)
    return worker.work()


# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class Worker():
    def __init__(self, name, should_stop):
        self.should_stop = should_stop
        self.name = "worker_" + str(name)
        self.number = name        
        self.summary_writer = tf.summary.create_file_writer('tensorboard/train_' + str(name))
        
        network_mod = __import__("deliverables.network", fromlist=[''])
        network_class = getattr(network_mod, 'AC_Network')

        curriculum_mod = __import__("deliverables.curriculum", fromlist=[''])
        curriculum_class =  getattr(curriculum_mod, 'Curriculum')
        self.curriculum = curriculum_class()

        self.params = __import__("deliverables.input_params", fromlist=[''])
        self.obs_helper = __import__("deliverables.observation", fromlist=[''])
        
        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_model = network_class(True,const.url, self.number)
        self.env = RailEnvWrapper(self.local_model.get_observation_builder())
        

    def train(self, rollout):
        all_rollouts = []
        all_rewards = []
        discounted_rewards = np.array([])

        for i in range(self.env.num_agents):
            rewards = [row[2] for row in rollout[i]]
            rewards = discount(rewards, self.params.gamma)
            all_rewards.append(rewards)
            all_rollouts += rollout[i]

        discounted_rewards = np.concatenate(all_rewards)
        actions = np.asarray([row[1] for row in all_rollouts]) 
        values = np.asarray([row[5] for row in all_rollouts])
        obs = self.obs_helper.buffer_to_obs_lists(all_rollouts)
        advantages = discounted_rewards - values

        v_l,p_l,e_l,g_n, v_n = self.local_model.train(discounted_rewards, advantages, actions, obs)
        return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n,  v_n
    

    def work(self):
        try:
            print ("Starting worker " + str(self.number))
            steps_on_level = 0
            self.episode_rewards = []
            self.episode_lengths = []
            self.episode_success = []
            self.episode_mean_values = []
            self.local_model.update_from_global_model()
            self.episode_count = 0

            while not bool(self.should_stop.value):
                # Check with server if there is a new curriculum level available
                if self.episode_count % 50 == 0:
                    self.curriculum.update_curriculum_level()
                    self.curriculum.update_env_to_curriculum_level(self.env)

                episode_done = False

                # Buffer for obs, action, next obs, reward
                episode_buffer = [[] for i in range(self.env.num_agents)]

                # A way to remember if the agent was already done during the last episode
                done_last_step = {i:False for i in range(self.env.num_agents)}

                # Metrics for tensorboard logging
                episode_values = []
                episode_reward = 0
                episode_step_count = 0
                
                obs, info = self.env.reset()

                while episode_done == False and episode_step_count < self.env.max_steps:
                    actions, v = self.local_model.get_actions_and_values(obs)
                    next_obs, rewards, done, info = self.env.step(actions)

                    episode_done = done['__all__']
                    if episode_done == True:
                        next_obs = obs

                    for i in range(self.env.num_agents):
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
                                v[i]])
                            
                            episode_values.append(v[i])
                            episode_reward += agent_reward
                    
                    obs = next_obs              
                    episode_step_count += 1
                    steps_on_level += 1
                    done_last_step = dict(done)

                # End of episode-loop
                if episode_done:
                    for i in range(self.env.num_agents):
                        # If agents could finish the level, 
                        # set final reward for all agents
                        episode_buffer[i][-1][2] += 40
                        episode_reward += 40
                else:
                    for i in range(self.env.num_agents):
                        if not done[i]:
                            episode_buffer[i][-1][2] -= 40
                            episode_reward -= 40

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))
                self.episode_success.append(episode_done)

                v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer)
                
                info[0] = v_l
                info[1] = p_l
                info[2] = e_l
                info[3] = g_n
                info[4] = v_n
                        
                # Save stats to Tensorboard every 5 episodes
                self.log_in_tensorboard(info)
                self.episode_count += 1
                print('Episode', self.episode_count,'of',self.name,'with',episode_step_count,'steps, reward of',episode_reward, ', mean entropy of', np.mean(info[2]), ', curriculum level ')
            
            return self.episode_count
    
        except KeyboardInterrupt:
            raise KeyboardInterruptError()


    def log_in_tensorboard(self, info):
        if self.episode_count % 5 == 0 and self.episode_count != 0:
            mean_length = np.mean(self.episode_lengths[-100:])
            mean_value = np.mean(self.episode_mean_values[-100:])
            mean_success_rate = np.mean(self.episode_success[-100:])
            mean_reward = np.mean(self.episode_rewards[-100:])

            with self.summary_writer.as_default():
                episode_count = np.int32(self.episode_count)
                tf.summary.scalar('Perf/Reward', mean_reward, step=episode_count)
                tf.summary.scalar('Perf/Length', mean_length, step=episode_count)
                tf.summary.scalar('Perf/Value', mean_value, step=episode_count)
                tf.summary.scalar('Perf/Successrate', mean_success_rate, step=episode_count)
                tf.summary.scalar('Losses/Value Loss', np.mean(info[0]), step=episode_count)
                tf.summary.scalar('Losses/Policy Loss', np.mean(info[1]), step=episode_count)
                tf.summary.scalar('Losses/Entropy', np.mean(info[2]), step=episode_count)
                tf.summary.scalar('Losses/Grad Norm', np.mean(info[3]), step=episode_count)
                tf.summary.scalar('Losses/Var Norm', np.mean(info[4]), step=episode_count)
                self.summary_writer.flush()




        