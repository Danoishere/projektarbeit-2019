import threading
import multiprocess as mp
import numpy as np
import tensorflow as tf
from ctypes import c_bool


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

def create_worker(name, checkpoint_manager, curriculum_manager, start_episode, lock, should_stop):
    worker = Worker(name, checkpoint_manager, curriculum_manager,start_episode, lock, should_stop)
    return worker.work()

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

class Worker():
    def __init__(self,name, checkpoint_manager, curriculum_manager, start_episode, lock, should_stop,):
        self.name = "worker_" + str(name)
        self.number = name        
        self.checkpoint_manager = checkpoint_manager
        self.curr_manager = curriculum_manager
        self.trainer = RMSprop(learning_rate=params.learning_rate)
        self.episode_count = start_episode
        self.should_stop = should_stop
        self.summary_writer = tf.summary.create_file_writer(const.tensorboard_path + 'train_' + str(name))
        
        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_model = AC_Network(self.trainer, const.model_path, True, True, lock)
        self.env = RailEnvWrapper(self.local_model.get_observation_builder())
        
    def train(self, rollout):
        all_rollouts = []
        all_rewards = []
        discounted_rewards = np.array([])

        for i in range(self.env.num_agents):
            rewards = [row[2] for row in rollout[i]]
            rewards = discount(rewards, params.gamma)
            all_rewards.append(rewards)
            all_rollouts += rollout[i]

        discounted_rewards = np.concatenate(all_rewards)
        actions = np.asarray([row[1] for row in all_rollouts]) 
        values = np.asarray([row[5] for row in all_rollouts])

        obs0 = np.asarray([row[0][0] for row in all_rollouts])
        obs1 = np.asarray([row[0][1] for row in all_rollouts])
        obs2 = np.asarray([row[0][2] for row in all_rollouts])
        obs = [obs0, obs1, obs2]

        advantages = discounted_rewards - values

        v_l,p_l,e_l,g_n_a, g_n_c, v_n_a, v_n_c = self.local_model.train(discounted_rewards, advantages, actions, obs, const.model_path)
        return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n_a, g_n_c, v_n_a, v_n_c
    
    def work(self):
        print ("Starting worker " + str(self.number))

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_success = []
        self.episode_mean_values = []

        steps_on_level = 0

        # Let the curriculum manager update the level to the current difficulty
        self.curr_manager.update_env_to_curriculum_level(self.env)

        while not bool(self.should_stop.value):
            self.local_model.update_from_global_model(const.model_path)
            episode_done = False

            # Buffer for obs, action, next obs, reward
            episode_buffer = [[] for i in range(self.env.num_agents)]

            # A way to remember if the agent was already done during the last episode
            done_last_step = {i:False for i in range(self.env.num_agents)}

            # Metrics for tensorboard logging
            episode_values = []
            episode_reward = 0
            episode_step_count = 0
            info = np.zeros(7)
            
            obs = self.env.reset()

            while episode_done == False and episode_step_count < self.env.max_steps:
                #Take an action using probabilities from policy network output.
                actions, v = self.local_model.get_actions_and_values(obs, self.env.num_agents)
                next_obs, rewards, done = self.env.step(actions)

                episode_done = done['__all__']
                if episode_done == True:
                    next_obs = obs

                for i in range(self.env.num_agents):
                    agent_obs = [obs[0][i],obs[1][i],obs[2][i]]
                    agent_action = actions[i]
                    agent_reward = rewards[i]
                    agent_next_obs =  [next_obs[0][i],next_obs[1][i],next_obs[2][i]] #[i]

                    if not done_last_step[i]:
                        episode_buffer[i].append([
                            agent_obs,
                            agent_action,
                            agent_reward,
                            agent_next_obs,
                            episode_done,
                            v[i,0]])
                        
                        episode_values.append(v[i,0])
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

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_step_count)
            self.episode_mean_values.append(np.mean(episode_values))
            self.episode_success.append(episode_done)
            
            self.local_model.update_from_global_model(const.model_path)

            # Update the network using the episode buffer at the end of the episode.
            # if episode_done:
            v_l, p_l, e_l, g_n_a, g_n_c, v_n_a, v_n_c = self.train(episode_buffer)
            
            info[0] = v_l
            info[1] = p_l
            info[2] = e_l
            info[3] = g_n_a
            info[4] = g_n_c
            info[5] = v_n_a
            info[6] = v_n_c
                    
            # Save stats to Tensorboard every 5 episodes
            if self.episode_count % 5 == 0 and self.episode_count != 0:
                mean_length = np.mean(self.episode_lengths[-100:])
                mean_value = np.mean(self.episode_mean_values[-100:])
                mean_success_rate = np.mean(self.episode_success[-100:])
                mean_reward = np.mean(self.episode_rewards[-100:])
                
                if self.checkpoint_manager != None:
                    self.checkpoint_manager.try_save_model(self.local_model, self.episode_count, mean_reward, self.name)
                self.curr_manager.report_success_rate(mean_success_rate, self.name, steps_on_level)
                
                with self.summary_writer.as_default():
                    episode_count = np.int32(self.episode_count)
                    tf.summary.scalar('Perf/Reward', mean_reward, step=episode_count)
                    tf.summary.scalar('Perf/Length', mean_length, step=episode_count)
                    tf.summary.scalar('Perf/Value', mean_value, step=episode_count)
                    tf.summary.scalar('Perf/Successrate', mean_success_rate, step=episode_count)
                    tf.summary.scalar('Losses/Value Loss', np.mean(info[0]), step=episode_count)
                    tf.summary.scalar('Losses/Policy Loss', np.mean(info[1]), step=episode_count)
                    tf.summary.scalar('Losses/Entropy', np.mean(info[2]), step=episode_count)
                    tf.summary.scalar('Losses/Grad Norm-Policy', np.mean(info[3]), step=episode_count)
                    tf.summary.scalar('Losses/Grad Norm-Value', np.mean(info[4]), step=episode_count)
                    tf.summary.scalar('Losses/Var Norm-Policy', np.mean(info[5]), step=episode_count)
                    tf.summary.scalar('Losses/Var Norm-Value', np.mean(info[6]), step=episode_count)
                    self.summary_writer.flush()

            self.episode_count += 1
            print('Episode', self.episode_count,'of',self.name,'with',episode_step_count,'steps, reward of',episode_reward, ', mean entropy of', np.mean(info[2]), ', curriculum level ', self.curr_manager.current_level)
        
        return self.episode_count