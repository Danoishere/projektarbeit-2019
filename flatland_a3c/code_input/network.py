import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model,load_model

import numpy as np
import code_input.input_params as params

class AC_Network():
    def __init__(self, global_model, trainer, create_networks=True):
        self.global_model = global_model
        self.trainer = trainer

        if create_networks:
            self.actor_model = self.actor_network()
            self.critic_model = self.critic_network()

    def critic_network(self):
        input_map =  layers.Input(shape=list(params.map_state_size) ,dtype=tf.float32)
        input_grid = layers.Input(shape=list(params.grid_state_size),dtype=tf.float32)
        input_vector = layers.Input(shape=list(params.vector_state_size),dtype=tf.float32)
        input_tree = layers.Input(shape=list(params.tree_state_size),dtype=tf.float32)

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

        value = layers.Dense(1)(hidden)

        return Model(
            inputs=[
                input_map,
                input_grid,
                input_vector,
                input_tree
            ],
            outputs=value)

    def actor_network(self):
        input_map =  layers.Input(shape=list(params.map_state_size) ,dtype=tf.float32)
        input_grid = layers.Input(shape=list(params.grid_state_size),dtype=tf.float32)
        input_vector = layers.Input(shape=list(params.vector_state_size),dtype=tf.float32)
        input_tree = layers.Input(shape=list(params.tree_state_size),dtype=tf.float32)

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

        policy = layers.Dense(params.number_of_actions, activation='softmax')(hidden)

        return Model(
            inputs=[
                input_map,
                input_grid,
                input_vector,
                input_tree
            ],
            outputs=policy)

    def update_from(self, from_model):
        from_vars = from_model.actor_model.trainable_variables
        to_vars = self.actor_model.trainable_variables
        op_holder = []
        for from_var,to_var in zip(from_vars,to_vars):
            op_holder.append(to_var.assign(from_var))

        from_vars = from_model.critic_model.trainable_variables
        to_vars = self.critic_model.trainable_variables
        op_holder = []
        for from_var,to_var in zip(from_vars,to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder


    def save_model(self, model_path, suffix):
        self.actor_model.save(model_path+'/actor_model_' + suffix + '.h5')
        self.critic_model.save(model_path+'/critic_model_' + suffix + '.h5')
        print('New',suffix,'model saved')


    def load_model(self,model_path, suffix):
        self.actor_model = load_model(model_path+'/actor_model_' + suffix + '.h5')
        self.critic_model = load_model(model_path+'/critic_model_' + suffix + '.h5')


    def value_loss(self, target_v, value):
        return 0.5 * tf.reduce_mean(tf.square(target_v - tf.reshape(value,[-1])))
    

    def policy_loss(self, advantages, actions, policy):
        actions_onehot = tf.one_hot(actions, params.number_of_actions)
        responsible_outputs = tf.reduce_sum(policy * actions_onehot, [1])
        entropy = - tf.reduce_sum(policy * tf.math.log(policy + 1e-10))
        return -tf.reduce_sum(tf.math.log(responsible_outputs  + 1e-10)*advantages) - entropy * 0.05, entropy


    def train(self, target_v, advantages, actions, obs):
        # Value loss
        with tf.GradientTape() as t:
            value = self.critic_model(obs)
            v_loss = self.value_loss(target_v,value)

        v_local_vars = self.critic_model.trainable_variables
        gradients = t.gradient(v_loss, v_local_vars)
        var_norms = tf.linalg.global_norm(v_local_vars)
        gradients, grad_norms = tf.clip_by_global_norm(gradients, 3.0)
        global_vars = self.global_model.critic_model.trainable_variables
        apply_grads = self.trainer.apply_gradients(zip(gradients, global_vars))

        # Policy loss
        with tf.GradientTape() as t:
            policy = self.actor_model(obs)
            p_loss, entropy = self.policy_loss(advantages, actions, policy)

        p_local_vars = self.actor_model.trainable_variables
        gradients = t.gradient(p_loss, p_local_vars)
        var_norms = tf.linalg.global_norm(p_local_vars)
        gradients, grad_norms = tf.clip_by_global_norm(gradients, 3.0)
        global_vars = self.global_model.actor_model.trainable_variables
        apply_grads = self.trainer.apply_gradients(zip(gradients, global_vars))
        
        return v_loss, p_loss, entropy, grad_norms, var_norms

    def get_actions_and_values(self, obs, num_agents):
        predcition = self.actor_model.predict([obs[0],obs[1],obs[2],obs[3]])
        values = self.critic_model.predict([obs[0],obs[1],obs[2],obs[3]])
        actions = {}
        for i in range(num_agents):
            a_dist = predcition[i]
            a = np.random.choice([0,1,2,3,4], p = a_dist)
            actions[i] = a

        return actions, values

    def get_actions(self, obs, num_agents):
        predcition = self.actor_model.predict([obs[0],obs[1],obs[2],obs[3]],)
        actions = {}
        for i in range(num_agents):
            a_dist = predcition[0][i]
            a = np.random.choice([0,1,2,3,4], p = a_dist)
            actions[i] = a

        return actions

    def get_values(self, obs, num_agents):
        return self.critic_model.predict([obs[0],obs[1],obs[2],obs[3]])
        '''
        values = {}
        for i in range(num_agents):
            values[i] = predcition[i]

        return values
        '''