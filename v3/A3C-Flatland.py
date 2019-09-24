#!/usr/bin/env python
# coding: utf-8

# This iPython notebook includes an implementation of the [A3C algorithm](https://arxiv.org/pdf/1602.01783.pdf).
# 
# `tensorboard --logdir=worker_0:./train_0,worker_1:./train_1,worker_2:./train_2,worker_3:./train_3`
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
#get_ipython().run_line_magic('matplotlib', 'inline')
from helper import *

from random import choice
from time import sleep
from time import time


# In[19]:

from observation import RawObservation

from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool
from flatland.envs.observations import TreeObsForRailEnv, LocalObsForRailEnv, GlobalObsForRailEnv
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator
from flatland.core.grid.grid4_astar import a_star


from make_env import make_env
from gym.spaces import Discrete




# ### Helper Functions

# In[20]:


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def reshape_obs(agent_observations):
    observations = []
    num_agents = len(agent_observations)
    for i in range(num_agents):
        agent_obs = agent_observations[i]
        observations.append(agent_obs)
    observations = np.array(observations)
    observations = np.reshape(observations,(num_agents, s_size[0],s_size[1],s_size[2], 1))
    return observations



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


# ### Actor-Critic Network

# In[21]:


class AC_Network():
    def __init__(self,s_size,a_size,scope,trainer):
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None,s_size[0]],dtype=tf.float32)
            
            def network(input):
                hidden = layers.Dense(256, activation='relu')(input)
                hidden = layers.Dropout(0.1)(hidden)
                hidden = layers.Dense(128, activation='relu')(hidden)
                hidden = layers.Dropout(0.1)(hidden)
                hidden = layers.Dense(64, activation='relu')(hidden)
                hidden = layers.Dropout(0.1)(hidden)
                hidden = layers.Dense(8, activation='relu')(hidden)
                return hidden

            v = network(self.inputs)
            p = network(self.inputs)

            #Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(32,state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(p, [0])
            step_size = tf.shape(self.inputs)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 32])
            
            #Output layers for policy and value estimations
            self.policy = slim.fully_connected(p,a_size,
                activation_fn=tf.nn.softmax,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)
            self.value = slim.fully_connected(v,1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)
            
            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.math.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.math.log(self.responsible_outputs)*self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                #Get gradients from local network using local losses
                local_vars = tf.compat.v1.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
                
                #Apply local gradients to global network
                global_vars = tf.compat.v1.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))


# ### Worker Agent

# In[22]:


class Worker():
    def __init__(self,name,s_size,a_size,trainer,model_path,global_episodes):
        self.name = "worker_" + str(name)
        self.number = name        
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.compat.v1.summary.FileWriter("train_" + str(self.number))

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size, a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)        
        
        num_agents = 2

        env = make_env('simple_spread')
        np.random.seed(123)
        env.seed(123)

        self.actions = [0,1,2,3,4]
        self.env = env
        
    def train(self, rollout, observations, sess, gamma, bootstrap_value):
        ''' Gradient decent for a single agent'''
        rollout = np.array(rollout)

        actions = rollout[:,1]
        rewards = rollout[:,2]
        values = rollout[:,5]
        
        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {
            self.local_AC.target_v : discounted_rewards,
            self.local_AC.inputs : np.array(observations),
            self.local_AC.actions : actions,
            self.local_AC.advantages : advantages,
            self.local_AC.state_in[0] : self.batch_rnn_state[0],
            self.local_AC.state_in[1] : self.batch_rnn_state[1]
        }
        
        v_l,p_l,e_l,g_n,v_n, self.batch_rnn_state,_ = sess.run([
            self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.state_out,
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
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                num_of_done_agents = 0
                
                obs = self.env.reset()
                episode_done = False
                
                rnn_state = self.local_AC.state_init
                self.batch_rnn_state = rnn_state
                    
                while episode_done == False and episode_step_count < max_episode_length:
                    #Take an action using probabilities from policy network output.
                    a_dist,v,rnn_state = sess.run([
                        self.local_AC.policy,
                        self.local_AC.value,
                        self.local_AC.state_out], 
                        feed_dict={
                            self.local_AC.inputs : obs,
                            self.local_AC.state_in[0] : rnn_state[0],
                            self.local_AC.state_in[1] : rnn_state[1]
                        })

                    actions = []
                    one_hot_actions = np.zeros((self.env.n,5))
                    for i in range(self.env.n):
                        a = np.random.choice([0,1,2,3,4], p = a_dist[i])
                        one_hot_actions[i,a] = 1
                        actions.append(a)

                    next_obs, rewards, done, info = self.env.step(one_hot_actions)
                    
                    # Is episode finished?
                    episode_done = np.all(done)
                  
                    if episode_done == True:
                        next_obs = obs

                    for i in range(self.env.n):
                        agent_state = obs[i]
                        agent_action = actions[i]
                        agent_reward = rewards[i]
                        agent_next_obs = next_obs[i]
                        episode_frames.append(agent_state)
                        episode_buffer.append([
                            agent_state,
                            agent_action,
                            agent_reward,
                            agent_next_obs,
                            episode_done,
                            v[0,0]])
                        
                        episode_values.append(v[0,0])
                        episode_reward += agent_reward
                    
                    obs = next_obs                  
                    total_steps += 1
                    episode_step_count += 1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == 30 and episode_done != True and episode_step_count != max_episode_length - 1:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        #for i in range(self.env.num_agents):
                        v1 = sess.run(self.local_AC.value, 
                            feed_dict={
                                self.local_AC.inputs : episode_frames,
                                self.local_AC.state_in[0] : rnn_state[0],
                                self.local_AC.state_in[1] : rnn_state[1]})[0,0]
                        v_l,p_l,e_l,g_n,v_n = self.train(
                            episode_buffer, 
                            episode_frames,
                            sess,
                            gamma,
                            v1)
                        episode_buffer = []
                        episode_frames = []
                        sess.run(self.update_local_ops)
                    if episode_done == True:
                        break
                                            
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))
                
                # Update the network using the episode buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    v_l,p_l,e_l,g_n,v_n = self.train(
                        episode_buffer,
                        episode_frames,
                        sess,
                        gamma,
                        0.0)
                    
                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if self.name == 'worker_0' and episode_count % 25 == 0:
                        '''
                        time_per_step = 0.05
                        images = np.array(episode_frames)
                        
                        make_gif(images,'./frames/image'+str(episode_count)+'.gif',
                            duration=len(images)*time_per_step,true_image=True,salience=False)
                        '''
                    if episode_count % 50 == 0 and self.name == 'worker_0':
                        saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print ("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)
                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1
                print('Episode', episode_count,'of',self.name,'with',episode_step_count,'steps')


# In[23]:


max_episode_length = 200
gamma = .99 # discount rate for advantage estimation and reward discounting
s_size = (18,) #  Observations are 21*21 with five channels
a_size = 5 # Agent can move Left, Right, or Fire
load_model = False
model_path = './model'


# In[24]:


tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)
    
#Create a directory to save episode playback gifs to
if not os.path.exists('./frames'):
    os.makedirs('./frames')

with tf.device("/cpu:0"): 
    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    master_network = AC_Network(s_size,a_size,'global',None) # Generate global network
    num_workers = multiprocessing.cpu_count() # Set workers to number of available CPU threads
    workers = []
    # Create worker classes
    for i in range(num_workers):
        workers.append(Worker(i,s_size,a_size,trainer,model_path,global_episodes))
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




