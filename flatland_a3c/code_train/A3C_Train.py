#!/usr/bin/env python
# coding: utf-8

# This iPython notebook includes an implementation of the [A3C algorithm](https://arxiv.org/pdf/1602.01783.pdf).
# 
# `tensorboard --logdir=worker_0:./train_0',worker_1:./train_1,worker_2:./train_2,worker_3:./train_3,worker_4:./train_4,worker_5:./train_5,worker_6:./train_6,worker_7:./train_7`
#
#  ##### Enable autocomplete


import threading
import multiprocessing
import numpy as np
import tensorflow as tf

import scipy.signal
from tensorflow.keras import layers, optimizers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model

from helper import *

from random import choice
from time import sleep
from time import time
from rail_env_wrapper import RailEnvWrapper
from code_input.network import AC_Network

import constants

# ### Helper Functions

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_model,to_model):
    from_vars = from_model.keras_model.trainable_variables# tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_model)
    to_vars = to_model.keras_model.trainable_variables # tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_model)
    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def max_length_sublist(top_list):
    max_len = 0
    for sub_list in top_list:
        list_len = len(sub_list)
        if list_len > max_len:
            max_len = list_len

    return max_len


class Worker():
    def __init__(self,name,global_model,trainer,model_path,global_episodes):
        self.name = "worker_" + str(name)
        self.number = name        
        self.model_path = model_path
        self.global_model = global_model
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_success = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.create_file_writer("train_" + str(self.number))
        self.rail_env_wrapper = RailEnvWrapper()

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_model = AC_Network(global_model, trainer)
        self.update_local_ops = lambda: update_target_graph(self.global_model, self.local_model)
        self.actions = [0,1,2,3,4]
        
    def train(self, rollout, gamma, bootstrap_value):
        ''' Gradient decent for a single agent'''

        observations_map = np.asarray([row[0][0] for row in rollout])
        observations_grid = np.asarray([row[0][1] for row in rollout])
        observations_vector = np.asarray([row[0][2] for row in rollout])
        observations_tree = np.asarray([row[0][3] for row in rollout])

        observations = [[row[0][0],row[0][1],row[0][2],row[0][3]] for row in rollout]

        actions = np.asarray([row[1] for row in rollout]) # rollout[:,1]
        rewards = np.asarray([row[2] for row in rollout]) # rollout[:,2]
        values = np.asarray([row[5] for row in rollout]) # rollout[:,5]
        
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
        '''
        feed_dict = {
            self.local_model.target_v : discounted_rewards,
            self.local_model.input_map : observations_map,
            self.local_model.input_grid: observations_grid,
            self.local_model.input_vector : observations_vector,
            self.local_model.input_tree : observations_tree,
            self.local_model.actions : actions,
            self.local_model.advantages : advantages
        }

        '''
        obs = [observations_map, observations_grid, observations_vector, observations_tree]
        v_l,p_l,e_l,g_n,v_n = self.local_model.train(discounted_rewards, advantages, actions, obs)

        '''
        
        v_l,p_l,e_l,g_n,v_n,_ = sess.run([
            self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)
        '''
        return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n,v_n
    
    def work(self,max_episode_length,gamma,coord):
        episode_count =  0 #self.global_episodes
        total_steps = 0
        print ("Starting worker " + str(self.number))
        while not coord.should_stop():
            self.update_local_ops()
            
            episode_done = False
            episode_buffers = [[] for i in range(self.rail_env_wrapper.num_agents)]
            done_last_step = {i:False for i in range(self.rail_env_wrapper.num_agents)}
            episode_values = []
            episode_reward = 0
            episode_step_count = 0
            
            obs = self.rail_env_wrapper.reset()
            info = np.zeros((self.rail_env_wrapper.num_agents,5))

            for i in range(self.rail_env_wrapper.num_agents):
                episode_buffers.append([])

            while episode_done == False and episode_step_count < max_episode_length:
                #Take an action using probabilities from policy network output.
                a_dist, v = self.local_model.keras_model.predict([obs[0],obs[1],obs[2],obs[3]])
                actions = {}
                for i in range(self.rail_env_wrapper.num_agents):
                    try:
                        a = np.random.choice([0,1,2,3,4], p = a_dist[i])
                        actions[i] = a
                    except:
                        import sys
                        np.set_printoptions(threshold=sys.maxsize)
                        print('Observations while error:')
                        print(obs)

                next_obs, rewards, done = self.rail_env_wrapper.step(actions)

                # Is episode finished?
                episode_done = done['__all__']
                
                if episode_done == True:
                    next_obs = obs

                for i in range(self.rail_env_wrapper.num_agents):
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

                # If the episode hasn't ended, but the experience buffer is full, then we
                # make an update step using that experience rollout.
                if max_length_sublist(episode_buffers) % 25 == 0 and not episode_done and episode_step_count != max_episode_length - 1:
                    # Since we don't know what the true final return is, we "bootstrap" from our current
                    # value estimation.
                    #for i in range(self.env.num_agents):

                    _,v1 = self.local_model.keras_model.predict(obs)

                    for i in range(self.rail_env_wrapper.num_agents):
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
                            self.update_local_ops()
                if episode_done == True:
                    break
                                        
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_step_count)
            self.episode_mean_values.append(np.mean(episode_values))
            self.episode_success.append(episode_done)
            
            # Update the network using the episode buffer at the end of the episode.
            if episode_done:
                for i in range(self.rail_env_wrapper.num_agents):
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
                        self.update_local_ops()
                    
            # Periodically save gifs of episodes, model parameters, and summary statistics.
            if episode_count % 5 == 0 and episode_count != 0:
                mean_reward = np.mean(self.episode_rewards[-100:])
                mean_length = np.mean(self.episode_lengths[-100:])
                mean_value = np.mean(self.episode_mean_values[-100:])
                mean_success_rate = np.mean(self.episode_success[-100:])

                if episode_count % 50 == 0 and self.name == 'worker_0':
                    #saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                    print ("Saved Checkpoint")
                
                if episode_count % 500 == 0 and self.name == 'worker_0':
                    self.local_model.keras_model.save(self.model_path+'/model-'+str(episode_count)+'.h5')
                    print ("Saved Keras Model")
                
                with self.summary_writer.as_default():
                    episode_count = np.int32(episode_count)
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
            if self.name == 'worker_0':
                #sess.run(self.increment)
                pass
            episode_count += 1
            print('Episode', episode_count,'of',self.name,'with',episode_step_count,'steps, reward of',episode_reward, ', mean entropy of', np.mean(info[:,2]))

# In[23]:

max_episode_length = 40
gamma = 0.98 # discount rate for advantage estimation and reward discounting

map_state_size = (11,11,9) #  Observations are 21*21 with five channels
grid_state_size = (11,11,16,1)
vector_state_size = 5
tree_state_size = 231

a_size = 5 # Agent can move Left, Right, or Fire
load_model = False
model_path = './model'

def start_train():
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    #Create a directory to save episode playback gifs to
    if not os.path.exists('./frames'):
        os.makedirs('./frames')

    with tf.device("/cpu:0"): 
        global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
        trainer = RMSprop(learning_rate=1e-4)
        master_network = AC_Network(None,None)
        num_workers = multiprocessing.cpu_count() 
        workers = []
        # Create worker classes
        for i in range(num_workers):
            workers.append(Worker(i,master_network,trainer,model_path,global_episodes))
        #saver = tf.train.Saver(max_to_keep=5)

    coord = tf.train.Coordinator()
    '''
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        pass
        #sess.run(tf.global_variables_initializer())
    '''
    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length,gamma,coord)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)
    print ("Looks like we're done")

if __name__ == "__main__":
    start_train()
