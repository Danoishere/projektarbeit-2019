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
    map_obs = []
    vec_obs = []
    num_agents = len(agent_observations)
    for i in range(num_agents):
        agent_obs = agent_observations[i]
        map_obs.append(agent_obs[0])
        vec_obs.append(agent_obs[1])
    map_obs = np.asarray(map_obs)
    map_obs = np.reshape(map_obs,(num_agents, s_size[0],s_size[1],s_size[2]))
    vec_obs = np.asarray(vec_obs)
    return [map_obs, vec_obs]

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
            #Input and visual encoding layers
            self.input_map = tf.placeholder(shape=[None,s_size[0],s_size[1],s_size[2]],dtype=tf.float32)
            self.input_vector = tf.placeholder(shape=[None,4],dtype=tf.float32)

            conv_policy = layers.Conv2D(64,(6,6))(self.input_map)
            conv_policy = layers.Conv2D(64,(3,3))(conv_policy)
            conv_policy = layers.Flatten()(conv_policy)

            flattend = layers.Flatten()(self.input_map)
            hidden_policy = layers.Dense(512, activation='relu')(flattend)
            hidden_policy = layers.Dropout(0.1)(hidden_policy)
            hidden_policy = layers.Dense(256, activation='relu')(hidden_policy)
            hidden_policy = layers.Dropout(0.1)(hidden_policy)
            hidden_policy = layers.Dense(64,activation='relu')(hidden_policy)
            hidden_policy = layers.concatenate([hidden_policy, self.input_vector,conv_policy])
            hidden_policy = layers.Dropout(0.1)(hidden_policy)
            hidden_policy = layers.Dense(64, activation='relu')(hidden_policy)
            hidden_policy = layers.Dropout(0.1)(hidden_policy)
            hidden_policy = layers.Dense(8, activation='relu')(hidden_policy)

            conv_value = layers.Conv2D(64,(6,6))(self.input_map)
            conv_value = layers.Conv2D(64,(3,3))(conv_value)
            conv_value = layers.Flatten()(conv_value)

            hidden_value = layers.Dense(512, activation='relu')(flattend)
            hidden_value = layers.Dropout(0.1)(hidden_value)
            hidden_value = layers.Dense(256,activation='relu')(hidden_value)
            hidden_value = layers.Dropout(0.1)(hidden_value)
            hidden_value = layers.Dense(64,activation='relu')(hidden_value)
            hidden_value = layers.concatenate([hidden_value, self.input_vector,conv_value])
            hidden_value = layers.Dropout(0.1)(hidden_value)
            hidden_value = layers.Dense(64, activation='relu')(hidden_value)
            hidden_value = layers.Dropout(0.1)(hidden_value)
            hidden_value = layers.Dense(8, activation='relu')(hidden_value)
            
            #Output layers for policy and value estimations
            self.policy = layers.Dense(a_size,activation='softmax')(hidden_policy)
            self.value = layers.Dense(1)(hidden_value)
            
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
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
                
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
        
        num_agents = 1
        rail_gen = complex_rail_generator(
            nr_start_goal=2,
            nr_extra=3,
            min_dist=10,
            seed=random.randint(0,100000)
        )
                    
        #The Below code is related to setting up the Flatland environment
        env = RailEnv(
                width=6,
                height=6,
                rail_generator = rail_gen,
                schedule_generator =complex_schedule_generator(),
                number_of_agents=num_agents,
                obs_builder_object=RawObservation([11,11]))

        env.step_penalty = -2
        env.global_reward = 10
        env.num_agents = num_agents

        self.actions = [0,1,2,3,4]
        self.env = env
        
    def train(self, rollout, sess, gamma, bootstrap_value):
        ''' Gradient decent for a single agent'''

        observations_map = np.asarray([row[0][0] for row in rollout])
        observations_vector = np.asarray([row[0][1] for row in rollout])
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
        #advantages = rewards + gamma * self.value_plus - self.value_plus
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {
            self.local_AC.target_v : discounted_rewards,
            self.local_AC.input_map : observations_map,
            self.local_AC.input_vector : observations_vector,
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
                episode_buffers = []
                episode_values = []
                episode_reward = 0
                episode_step_count = 0
                num_of_done_agents = 0
                
                obs = self.env.reset()
                obs = reshape_obs(obs)

                while obs[0].shape[0] == 0:
                    obs = self.env.reset()
                    obs = reshape_obs(obs)

                episode_done = False
                
                done_last_step = {}                
                dist = {}

                for i in range(self.env.num_agents):
                    episode_buffers.append([])
                    done_last_step[i] = False
                    dist[i] = 100
                    
                while episode_done == False and episode_step_count < max_episode_length:
                    #Take an action using probabilities from policy network output.
                    a_dist,v = sess.run([
                        self.local_AC.policy,
                        self.local_AC.value], 
                        feed_dict={
                            self.local_AC.input_map : obs[0],
                            self.local_AC.input_vector : obs[1]
                        })

                    actions = {}
                    for i in range(self.env.num_agents):
                        a = np.random.choice([0,1,2,3,4], p = a_dist[i])
                        actions[i] = a

                    next_obs, rewards, done, _ = self.env.step(actions)
                    next_obs = reshape_obs(next_obs)

                    num_of_done_agents = modify_reward(self.env, rewards, done, done_last_step, num_of_done_agents, dist)
                    
                    
                    # Is episode finished?
                    episode_done = done['__all__']
                  
                    if episode_done == True:
                        next_obs = obs

                    for i in range(self.env.num_agents):
                        agent_obs = [obs[0][i],obs[1][i]]
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
                    if len(episode_buffers[0]) == 25 and not episode_done and episode_step_count != max_episode_length - 1:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        #for i in range(self.env.num_agents):

                        v1 = sess.run(self.local_AC.value, 
                            feed_dict={
                                self.local_AC.input_map : obs[0],
                                self.local_AC.input_vector : obs[1]
                            })

                        info = np.zeros((self.env.num_agents,5))
                        for i in range(self.env.num_agents):
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
                
                # Update the network using the episode buffer at the end of the episode.
                if episode_done and len(episode_buffers[0]) != 0:
                    info = np.zeros((self.env.num_agents,5))
                    for i in range(self.env.num_agents):
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
                print('Episode', episode_count,'of',self.name,'with',episode_step_count,'steps, reward of',episode_reward)

# In[23]:


max_episode_length = 80
gamma = 0.98 # discount rate for advantage estimation and reward discounting
s_size = (11,11,23) #  Observations are 21*21 with five channels
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




