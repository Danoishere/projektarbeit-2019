#!/usr/bin/env python
# coding: utf-8

# This iPython notebook includes an implementation of the [A3C algorithm](https://arxiv.org/pdf/1602.01783.pdf).
# 
# tensorboard --logdir=deliverables/tensorboard
#
#  ##### Enable autocomplete


import threading
import multiprocessing
import numpy as np
import tensorflow as tf

import scipy.signal
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model

from datetime import datetime
from random import choice
from time import sleep
from time import time
from rail_env_wrapper import RailEnvWrapper
from code_util.checkpoint import CheckpointManager
from code_train.curriculum import CurriculumManager
from deliverables.network import AC_Network

import code_util.constants as const
import deliverables.input_params as params

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def max_length_sublist(top_list):
    max_len = 0
    for sub_list in top_list:
        list_len = len(sub_list)
        if list_len > max_len:
            max_len = list_len

    return max_len

class Worker():
    def __init__(self,name,global_model,trainer,checkpoint_manager, curriculum_manager, start_episode):
        self.name = "worker_" + str(name)
        self.number = name        
        self.checkpoint_manager = checkpoint_manager
        self.curr_manager = curriculum_manager
        self.global_model = global_model
        self.trainer = trainer
        self.episode_count = start_episode
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_success = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.create_file_writer(const.tensorboard_path + 'train_' + str(name))
        
        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_model = AC_Network(global_model, trainer)
        self.update_local_model = lambda: self.local_model.update_from(self.global_model)
        self.env = RailEnvWrapper(self.local_model.get_observation_builder())
        
    def train(self, rollout, gamma, bootstrap_value):
        ''' Gradient decent for a SINGLE agent'''

        observations_map = np.asarray([row[0][0] for row in rollout])
        observations_grid = np.asarray([row[0][1] for row in rollout])
        observations_vector = np.asarray([row[0][2] for row in rollout])
        observations_tree = np.asarray([row[0][3] for row in rollout])
        
        obs = [observations_map, observations_grid, observations_vector, observations_tree]

        actions = np.asarray([row[1] for row in rollout]) 
        rewards = np.asarray([row[2] for row in rollout])
        values = np.asarray([row[5] for row in rollout])
        
        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(np.append(rewards, bootstrap_value))
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(np.append(values,bootstrap_value))
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        v_l,p_l,e_l,g_n,v_n = self.local_model.train(discounted_rewards, advantages, actions, obs)

        return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n,v_n
    
    def work(self, max_episode_length, gamma, coord):
        total_steps = 0
        print ("Starting worker " + str(self.number))

        # Let the curriculum manager update the level to the current difficulty
        self.curr_manager.update_env_to_curriculum_level(self.env)

        while not coord.should_stop():
            self.update_local_model()
            episode_done = False

            # Buffer for obs, action, next obs, reward
            episode_buffers = [[] for i in range(self.env.num_agents)]

            # A way to remember if the agent was already done during the last episode
            done_last_step = {i:False for i in range(self.env.num_agents)}

            # Metrics for tensorboard logging
            episode_values = []
            episode_reward = 0
            episode_step_count = 0
            info = np.zeros((self.env.num_agents,5))
            
            obs = self.env.reset()
            
            while episode_done == False and episode_step_count < max_episode_length:
                #Take an action using probabilities from policy network output.
                actions, v = self.local_model.get_actions_and_values(obs, self.env.num_agents)
                next_obs, rewards, done = self.env.step(actions)

                episode_done = done['__all__']
                if episode_done == True:
                    next_obs = obs

                for i in range(self.env.num_agents):
                    agent_obs = [obs[0][i],obs[1][i],obs[2][i],obs[3][i]]

                    agent_action = actions[i]
                    agent_reward = rewards[i]
                    agent_next_obs = next_obs[i]

                    if not done_last_step[i]:
                        episode_buffers[i].append([
                            agent_obs,
                            agent_action,
                            agent_reward,
                            agent_next_obs,
                            episode_done,
                            v[i,0]])
                        
                        episode_values.append(v[i,0])
                        episode_reward += agent_reward
                
                obs = next_obs                  
                total_steps += 1
                episode_step_count += 1
                done_last_step = dict(done)

                '''
                # If the episode hasn't ended, but the experience buffer is full, then we
                # make an update step using that experience rollout.
                if max_length_sublist(episode_buffers) % 25 == 0 and not episode_done and episode_step_count != max_episode_length - 1:
                    # Since we don't know what the true final return is, we "bootstrap" from our current
                    # value estimation.
                    #for i in range(self.env.num_agents):

                    v1 = self.local_model.get_values(obs, self.env.num_agents)

                    for i in range(self.env.num_agents):
                        if len(episode_buffers[i]) > 0:
                            v_l,p_l,e_l,g_n,v_n = self.train(
                                episode_buffers[i], 
                                gamma,
                                v1[i,0])
                                
                            info[i,0] = v_l
                            info[i,1] = p_l
                            info[i,2] = e_l
                            info[i,3] = g_n
                            info[i,4] = v_n

                            episode_buffers[i] = []
                            self.update_local_model()
                '''
            # End of episode-loop

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_step_count)
            self.episode_mean_values.append(np.mean(episode_values))
            self.episode_success.append(episode_done)
            
            # Update the network using the episode buffer at the end of the episode.
            # if episode_done:
            for i in range(self.env.num_agents):
                if len(episode_buffers[i]) != 0:
                    v_l,p_l,e_l,g_n,v_n = self.train(
                        episode_buffers[i],
                        gamma,
                        0.0)
                    
                    info[i,0] = v_l
                    info[i,1] = p_l
                    info[i,2] = e_l
                    info[i,3] = g_n
                    info[i,4] = v_n
            
            self.update_local_model()
                    
            # Save stats to Tensorboard every 5 episodes
            if self.episode_count % 5 == 0 and self.episode_count != 0:
                mean_length = np.mean(self.episode_lengths[-100:])
                mean_value = np.mean(self.episode_mean_values[-100:])
                mean_success_rate = np.mean(self.episode_success[-100:])
                mean_reward = np.mean(self.episode_rewards[-100:])
                
                self.checkpoint_manager.try_save_model(self.episode_count, mean_reward, self.name)
                self.curr_manager.report_success_rate(mean_success_rate, self.name)
                
                with self.summary_writer.as_default():
                    episode_count = np.int32(self.episode_count)
                    tf.summary.scalar('Perf/Reward', mean_reward, step=episode_count)
                    tf.summary.scalar('Perf/Length', mean_length, step=episode_count)
                    tf.summary.scalar('Perf/Value', mean_value, step=episode_count)
                    tf.summary.scalar('Perf/Successrate/100 episodes', mean_success_rate, step=episode_count)
                    tf.summary.scalar('Losses/Value Loss', np.mean(info[:,0]), step=episode_count)
                    tf.summary.scalar('Losses/Policy Loss', np.mean(info[:,1]), step=episode_count)
                    tf.summary.scalar('Losses/Entropy', np.mean(info[:,2]), step=episode_count)
                    tf.summary.scalar('Losses/Grad Norm', np.mean(info[:,3]), step=episode_count)
                    tf.summary.scalar('Losses/Var Norm', np.mean(info[:,4]), step=episode_count)
                    self.summary_writer.flush()

            self.episode_count += 1
            print('Episode', self.episode_count,'of',self.name,'with',episode_step_count,'steps, reward of',episode_reward, ', mean entropy of', np.mean(info[:,2]), ', curriculum level ', self.curr_manager.current_level)

def start_train(resume):
    trainer = RMSprop(learning_rate=params.learning_rate)
    global_model = AC_Network(None,None)
    num_workers = min([multiprocessing.cpu_count(),const.max_workers])
    coord = tf.train.Coordinator()

    # Curriculum-manager manages the generation of the levels
    curr_manager = CurriculumManager(coord, 'worker_0')

    # Checkpoint-manager saves model-checkpoints
    ckpt_manager = CheckpointManager(global_model, curr_manager, 'worker_0', save_best_after_min=30, save_ckpt_after_min=100)

    start_episode = 0
    if resume == True:
        print ('Loading Model...')
        start_episode = ckpt_manager.load_checkpoint_model()

    workers = []
    for i in range(num_workers):
        workers.append(Worker(i,global_model,trainer,ckpt_manager, curr_manager, start_episode))

    worker_threads = []
    while not curr_manager.stop_training:
        for worker in workers:
            worker_work = lambda: worker.work(params.max_episode_length,params.gamma,coord)
            t = threading.Thread(target=(worker_work))
            t.start()
            sleep(0.5)
            worker_threads.append(t)

        coord.join(worker_threads)
        coord.clear_stop()

        # Save model after each curriculum level
        run_time = datetime.now().strftime("%m/%d/%Y_%H:%M:%S")
        global_model.save_model(const.model_path_hist,'level_' + str(curr_manager.current_level) + '_' + run_time)
        curr_manager.switch_to_next_level()

    print ("Looks like we're done")

if __name__ == "__main__":
    start_train(False)
