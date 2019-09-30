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
import tensorflow.contrib.slim as slim
import scipy.signal
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from helper import *

from random import choice
from time import sleep
from time import time
from rail_env_wrapper import Rail_Env_Wrapper

import constants


# ### Helper Functions

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
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




class AC_Network():
    def __init__(self,a_size,scope,trainer):
        with tf.variable_scope(scope):
            self.input_map =  layers.Input(shape=[map_state_size[0],map_state_size[1],map_state_size[2]],dtype=tf.float32)
            self.input_grid = layers.Input(shape=[grid_state_size[0],grid_state_size[1],grid_state_size[2],1],dtype=tf.float32)
            self.input_vector = layers.Input(shape=[vector_state_size],dtype=tf.float32)
            self.input_tree = layers.Input(shape=[tree_state_size],dtype=tf.float32)

            def network(input_map,input_grid,input_vector, input_tree):
                conv_grid = layers.Conv3D(32,(1,1,4),strides=(1,1,4))(input_grid)
                conv_grid = layers.Flatten()(conv_grid)
                conv_grid_hidden = layers.Dense(64, activation='relu')(conv_grid)

                conv_map = layers.Conv2D(32,(3,3))(input_map)
                conv_map = layers.Flatten()(conv_map)
                conv_map_hidden = layers.Dense(64, activation='relu')(conv_map)

                flattend_map = layers.Flatten()(input_map)
                hidden_map = layers.Dense(128, activation='relu')(flattend_map)

                hidden_tree = layers.BatchNormalization()(input_tree)
                hidden_tree = layers.Dense(128, activation='relu')(hidden_tree)
                hidden_vector = layers.Dense(32, activation='relu')(input_vector)

                hidden = layers.concatenate([hidden_map, hidden_vector, conv_grid_hidden, conv_map_hidden, hidden_tree])
                hidden = layers.Dense(256, activation='relu')(hidden)
                hidden = layers.Dropout(0.2)(hidden)
                hidden = layers.Dense(32, activation='relu')(hidden)
                return hidden

            out_policy = network(self.input_map,self.input_grid,self.input_vector, self.input_tree)
            out_value = network(self.input_map,self.input_grid,self.input_vector, self.input_tree)

            #Output layers for policy and value estimations
            self.policy = layers.Dense(a_size,activation='softmax')(out_policy)
            self.value = layers.Dense(1)(out_value)

            self.keras_model = Model(
                inputs=[
                    self.input_map,
                    self.input_grid,
                    self.input_vector,
                    self.input_tree
                ],
                outputs=[
                    self.policy,
                    self.value
                ])

            if scope != 'global':
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)
                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.math.log(self.policy + 1e-10))
                self.policy_loss = -tf.reduce_sum(tf.math.log(self.responsible_outputs  + 1e-10)*self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.05

                #Get gradients from local network using local losses
                local_vars = tf.compat.v1.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.linalg.global_norm(local_vars)
                self.gradients, self.grad_norms = tf.clip_by_global_norm(self.gradients, 5.0)

                #Apply local gradients to global network
                global_vars = tf.compat.v1.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(self.gradients, global_vars))


class Worker():
    def __init__(self,name,a_size,trainer,model_path,global_episodes):
        self.name = "worker_" + str(name)
        self.number = name        
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_success = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))
        self.rail_env_wrapper = Rail_Env_Wrapper()

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)

        self.actions = [0,1,2,3,4]
        
    def train(self, rollout, sess, gamma, bootstrap_value):
        ''' Gradient decent for a single agent'''

        observations_map = np.asarray([row[0][0] for row in rollout])
        observations_grid = np.asarray([row[0][1] for row in rollout])
        observations_vector = np.asarray([row[0][2] for row in rollout])
        observations_tree = np.asarray([row[0][3] for row in rollout])

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
        feed_dict = {
            self.local_AC.target_v : discounted_rewards,
            self.local_AC.input_map : observations_map,
            self.local_AC.input_grid: observations_grid,
            self.local_AC.input_vector : observations_vector,
            self.local_AC.input_tree : observations_tree,
            self.local_AC.actions : actions,
            self.local_AC.advantages : advantages
        }
        
        v_l,p_l,e_l,g_n,v_n,_ = sess.run([
            self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)
        
        return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n,v_n
    
    def work(self,max_episode_length,gamma,sess,coord,saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        
        print ("Starting worker " + str(self.number))
        
        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                
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
                    a_dist,v = sess.run([
                        self.local_AC.policy,
                        self.local_AC.value], 
                        feed_dict={
                            self.local_AC.input_map : obs[0],
                            self.local_AC.input_grid: obs[1],
                            self.local_AC.input_vector : obs[2],
                            self.local_AC.input_tree : obs[3]
                        })

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

                        v1 = sess.run(self.local_AC.value, 
                            feed_dict={
                                self.local_AC.input_map : obs[0],
                                self.local_AC.input_grid: obs[1],
                                self.local_AC.input_vector : obs[2],
                                self.local_AC.input_tree : obs[3]
                            })

                        for i in range(self.rail_env_wrapper.num_agents):
                            if len(episode_buffers[i]) > 0:
                                v_l,p_l,e_l,g_n,v_n = self.train(
                                    episode_buffers[i], 
                                    sess,
                                    gamma,
                                    v1[i,0])
                                    
                                info[i,0] = v_l
                                info[i,1] = p_l
                                info[i,2] = e_l
                                info[i,3] = g_n
                                info[i,4] = v_n

                                episode_buffers[i] = []
                                sess.run(self.update_local_ops)
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
                                sess,
                                gamma,
                                0.0)
                            
                            info[i,0] = v_l
                            info[i,1] = p_l
                            info[i,2] = e_l
                            info[i,3] = g_n
                            info[i,4] = v_n
                            sess.run(self.update_local_ops)
                        
                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    mean_reward = np.mean(self.episode_rewards[-100:])
                    mean_length = np.mean(self.episode_lengths[-100:])
                    mean_value = np.mean(self.episode_mean_values[-100:])
                    mean_success_rate = np.mean(self.episode_success[-100:])

                    if episode_count % 50 == 0 and self.name == 'worker_0':
                        saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print ("Saved Checkpoint")
                    
                    if episode_count % 500 == 0 and self.name == 'worker_0':
                        self.local_AC.keras_model.save(self.model_path+'/model-'+str(episode_count)+'.h5')
                        print ("Saved Keras Model")
                    
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Perf/Successrate/100 episodes', simple_value=float(mean_success_rate))

                    summary.value.add(tag='Losses/Value Loss', simple_value=float(np.mean(info[:,0])))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(np.mean(info[:,1])))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(np.mean(info[:,2])))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(np.mean(info[:,3])))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(np.mean(info[:,4])))
                    self.summary_writer.add_summary(summary, episode_count)
                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1
                print('Episode', episode_count,'of',self.name,'with',episode_step_count,'steps, reward of',episode_reward, ', mean entropy of', np.mean(info[:,2]))

# In[23]:

max_episode_length = 40
gamma = 0.98 # discount rate for advantage estimation and reward discounting

map_state_size = (11,11,9) #  Observations are 21*21 with five channels
grid_state_size = (11,11,16)
vector_state_size = 5
tree_state_size = 231

a_size = 5 # Agent can move Left, Right, or Fire
load_model = False
model_path = './model'

def start_train():
    tf.reset_default_graph()

    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    #Create a directory to save episode playback gifs to
    if not os.path.exists('./frames'):
        os.makedirs('./frames')

    with tf.device("/cpu:0"): 
        global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
        trainer = optimizers.RMSprop(learning_rate=1e-4, clipnorm=1.0)
        master_network = AC_Network(a_size,'global',None)
        num_workers = multiprocessing.cpu_count() 
        workers = []
        # Create worker classes
        for i in range(num_workers):
            workers.append(Worker(i,a_size,trainer,model_path,global_episodes))
        saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if load_model == True:
            print ('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
            
        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate threat.
        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
            t = threading.Thread(target=(worker_work))
            t.start()
            sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)
        print ("Looks like we're done")

if __name__ == "__main__":
    start_train()
