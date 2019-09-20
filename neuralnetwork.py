from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D,concatenate

import numpy as np
import tensorflow as tf
import constants as const

def convert_global_obs(global_obs):
    observations = []
    num_agents = len(global_obs)
    for i in range(num_agents):
        agent_obs = global_obs[i]
        state1 = np.array(agent_obs[0])
        state2 = np.array(agent_obs[1])
        observation1 = state1.astype(np.float).reshape((1,state1.shape[0],state1.shape[1],state1.shape[2]))
        observation2 = state2.astype(np.float).reshape((1,state2.shape[0]))
        observations.append([observation1, observation2])

    return observations

def single_obs_to_tensor(observation):
    obs1 = tf.convert_to_tensor(observation[0], dtype=tf.float32)
    obs2 = tf.convert_to_tensor(observation[1], dtype=tf.float32)
    return [obs1,obs2]

def obs_list_to_tensor(observations, skip_index = False):
    t_obs1 = []
    t_obs2 = []
    for i in range(len(observations)):
        observation = observations[i]
        if skip_index:
            t_obs1.append(observation[0])
            t_obs2.append(observation[1])
        else:
            t_obs1.append(observation[0][0])
            t_obs2.append(observation[1][0])

    t_obs1 = tf.convert_to_tensor(t_obs1, dtype=tf.float32)
    t_obs2 = tf.convert_to_tensor(t_obs2, dtype=tf.float32)
    return [t_obs1,t_obs2]

def create_model():
    batch_size=10
    i1 = Input(shape=(21,21,6),batch_size=batch_size)
    c = Conv2D(32, kernel_size=(5,5))(i1)
    c = MaxPooling2D()(c)
    c = Conv2D(32, kernel_size=(4,4))(c)
    c = MaxPooling2D()(c)
    f1 = Flatten()(c)

    res_i1 = Flatten()(i1)

    i2 = Input(shape=(6,),batch_size=batch_size)
    r = Dense(64)(i2)
    r = Dense(64)(r)
    r = Flatten()(r)
    i = concatenate([f1,r])
    
    # Value network
    v = Dense(300, activation='relu')(i)
    v = Dense(300, activation='relu')(v)
    v = concatenate([v,res_i1,i2])
    v = Dense(300, activation='relu')(v)
    v = Dense(200, activation='relu')(v)
    v = Dense(100, activation='relu')(v)
    value = Dense(1)(v)

    c = Conv2D(32, kernel_size=(5,5))(i1)
    c = MaxPooling2D()(c)
    c = Conv2D(32, kernel_size=(4,4))(c)
    c = MaxPooling2D()(c)
    f1 = Flatten()(c)

    r = Dense(64)(i2)
    r = Dense(64)(r)
    r = Flatten()(r)
    i = concatenate([f1,r])

    # Policy network
    p = Dense(300, activation='relu')(i)
    p = Dense(300, activation='relu')(p)
    p = concatenate([p,res_i1,i2])
    p = Dense(300, activation='relu')(p)
    p = Dense(200, activation='relu')(p)
    p = Dense(100, activation='relu')(p)
    policy = Dense(const.ACTION_SIZE,activation='softmax')(p)

    model = Model(inputs=[i1,i2], outputs=[policy,value])
    return model