import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model

import numpy as np
import code_input.input_params as params

class AC_Network():
    def __init__(self, global_model, trainer):
        self.global_model = global_model
        self.trainer = trainer
        self.input_map =  layers.Input(shape=list(params.map_state_size) ,dtype=tf.float32)
        self.input_grid = layers.Input(shape=list(params.grid_state_size),dtype=tf.float32)
        self.input_vector = layers.Input(shape=list(params.vector_state_size),dtype=tf.float32)
        self.input_tree = layers.Input(shape=list(params.tree_state_size),dtype=tf.float32)

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
        self.policy = layers.Dense(params.number_of_actions, activation='softmax')(out_policy)
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


        #if trainer != None:
        #    self.keras_model.compile(trainer, loss=[self.policy_loss, self.value_loss])

        '''
        if global_model != None:
            self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions,params.number_of_actions,dtype=tf.float32)
            self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
            self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)
            self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

            #Loss functions
            self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
            self.entropy = - tf.reduce_sum(self.policy * tf.math.log(self.policy + 1e-10))
            self.policy_loss = -tf.reduce_sum(tf.math.log(self.responsible_outputs  + 1e-10)*self.advantages)
            self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.05

            #Get gradients from local network using local losses
            local_vars = self.keras_model.trainable_variables
            self.gradients = tf.gradients(self.loss, local_vars)
            self.var_norms = tf.linalg.global_norm(local_vars)
            self.gradients, self.grad_norms = tf.clip_by_global_norm(self.gradients, 5.0)

            #Apply local gradients to global network
            global_vars = self.global_model.keras_model.trainable_variables
            self.apply_grads = trainer.apply_gradients(zip(self.gradients, global_vars))
            '''

    def value_loss(self, target_v, value):
        return 0.5 * tf.reduce_mean(tf.square(target_v - tf.reshape(value,[-1])))
    
    def policy_loss(self, advantages, actions, policy):
        actions_onehot = tf.one_hot(actions, params.number_of_actions)
        responsible_outputs = tf.reduce_sum(policy * actions_onehot, [1])
        entropy = - tf.reduce_sum(policy * tf.math.log(policy + 1e-10))
        return -tf.reduce_sum(tf.math.log(responsible_outputs  + 1e-10)*advantages) - entropy * 0.05, entropy

    def train(self, target_v, advantages, actions, obs):
        with tf.GradientTape() as t:
            policy, value = self.keras_model(obs)
            v_loss = self.value_loss(target_v,value)
            p_loss, entropy = self.policy_loss(advantages, actions,policy)
            loss = 0.5 * v_loss + p_loss

        local_vars = self.keras_model.trainable_variables
        gradients = t.gradient(loss, local_vars)
        var_norms = tf.linalg.global_norm(local_vars)
        gradients, grad_norms = tf.clip_by_global_norm(gradients, 5.0)

        #Apply local gradients to global network
        global_vars = self.global_model.keras_model.trainable_variables
        apply_grads = self.trainer.apply_gradients(zip(gradients, global_vars))

        return v_loss, p_loss, entropy, grad_norms, var_norms


    def get_actions(self, obs, num_agents):
        predcition = self.keras_model.predict([obs[0],obs[1],obs[2],obs[3]])
        actions = {}
        for i in range(num_agents):
            a_dist = predcition[0][i]
            a = np.random.choice([0,1,2,3,4], p = a_dist)
            actions[i] = a

        return actions

    def get_values(self, obs, num_agents):
        predcition = self.keras_model.predict([obs[0],obs[1],obs[2],obs[3]])
        values = {}
        for i in range(num_agents):
            values[i] = predcition[1][i]

        return values