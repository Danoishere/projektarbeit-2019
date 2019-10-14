import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model,load_model

import numpy as np
import requests 
import dill
import deliverables.input_params as params
import json_tricks as json

from io import StringIO
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from deliverables.observation import CombinedObservation

import base64
import hashlib
import urllib


class AC_Network():
    def __init__(self, create_network=True, global_model_url = '', name = ''):
        self.name = str(name)
        self.global_model_url = global_model_url
        self.model = self.build_network()
        self.network_hash = self.get_network_hash()
        
        

    def get_network_hash(self):
        summary = StringIO()
        self.model.summary(print_fn=lambda x: summary.write(x + '\n'))
        summary = summary.getvalue().encode('utf-8')
        hasher = hashlib.sha1(summary)
        nw_hash = base64.urlsafe_b64encode(hasher.digest()[:10])
        return str(nw_hash, 'utf-8')


    def build_network(self):
        input_map = layers.Input(shape=list(params.map_state_size),dtype=tf.float32)
        input_grid = layers.Input(shape=list(params.grid_state_size),dtype=tf.float32)
        input_vec_tree = layers.Input(shape=list(params.vec_tree_state_size),dtype=tf.float32)

        actor_out = self.create_network(input_map, input_grid, input_vec_tree)
        critic_out = self.create_network(input_map, input_grid, input_vec_tree)

        policy = layers.Dense(params.number_of_actions, activation='softmax')(actor_out)
        value = layers.Dense(1)(critic_out)

        return Model(
            inputs=[input_map, input_grid, input_vec_tree],
            outputs=[policy, value])

    def create_network(self, input_map, input_grid, input_vec_tree):
        map_dense = layers.Flatten()(input_map)
        map_dense = layers.Dense(128, activation='relu')(map_dense)
        map_dense = layers.Dense(64, activation='relu')(map_dense)

        map_conv = layers.Conv2D(64,(4,4), activation='relu')(input_map)
        map_conv = layers.Dense(32, activation='relu')(map_conv)
        map_conv = layers.Flatten()(map_conv)
        map_conv = layers.Dense(16, activation='relu')(map_conv)

        grid_conv = layers.Conv2D(64, (3,3), activation='relu')(input_grid)
        grid_dense = layers.Flatten()(grid_conv)
        grid_dense = layers.Dense(64)(grid_dense)
        
        tree_dense = layers.Dense(256, activation='relu')(input_vec_tree)
        tree_dense = layers.Dense(64, activation='relu')(input_vec_tree)

        hidden = layers.concatenate([map_dense, grid_dense, tree_dense, map_conv])
        hidden = layers.Dense(128, activation='relu')(hidden)
        hidden = layers.Dense(32, activation='relu')(hidden)
        hidden = layers.Dense(8, activation='relu')(hidden)

        return hidden


    def update_from_global_model(self):
        ''' Updates the local copy of the global model 
        '''
        urllib.request.urlretrieve(self.global_model_url + '/get_global_weights', 'deliverables/weights_'+self.name+'.h5')
        self.model.load_weights('deliverables/weights_'+self.name+'.h5')


    def save_model(self, model_path, suffix):
        self.model.save(model_path+'/model_' + suffix + '.h5')
        print('New',suffix,'model saved')


    def load_model(self,model_path, suffix):
        self.model = load_model(model_path+'/model_' + suffix + '.h5')


    def value_loss(self, rec_reward, est_reward):
        return 0.5 * tf.reduce_sum(tf.square(rec_reward - tf.reshape(est_reward,[-1])))
    

    def policy_loss(self, advantages, actions, policy):
        actions_onehot = tf.one_hot(actions, params.number_of_actions)
        responsible_outputs = tf.reduce_sum(policy * actions_onehot, [1])
        policy_log = tf.math.log(tf.clip_by_value(policy, 1e-10, 1.0))
        entropy = - tf.reduce_sum(policy * policy_log)
        policy_loss = -tf.reduce_mean(tf.math.log(responsible_outputs  + 1e-10)*advantages) - entropy * params.entropy_factor
        return policy_loss, entropy


    def train(self, target_v, advantages, actions, obs):

        # Value loss
        with tf.GradientTape() as tape:
            policy,value = self.model(obs)
            v_loss = self.value_loss(target_v, value)
            p_loss, entropy = self.policy_loss(advantages, actions, policy)
            tot_loss = p_loss + v_loss

        local_vars = self.model.trainable_variables
        gradients = tape.gradient(tot_loss, local_vars)
        var_norms = tf.linalg.global_norm(local_vars)
        gradients, grad_norms = tf.clip_by_global_norm(gradients, params.gradient_norm)
        gradients_str = dill.dumps(gradients)

        # Send gradient update and receive new global weights
        data_resp = requests.post(
            url=self.global_model_url + '/send_gradient', 
            data=gradients_str)

        self.update_from_global_model()
        return v_loss, p_loss, entropy, grad_norms, var_norms


    def sigmoid(self,x, derivative=False):
        return x*(1-x) if derivative else 1/(1+np.exp(-x))


    def reshape_obs(self, obs):
        tree_obs = np.asarray(obs['obs1'])
        tree_obs[tree_obs ==  np.inf] = -1
        tree_obs[tree_obs ==  -np.inf] = -2
        tree_obs += 2
        tree_obs /= 40.0
        tree_obs = tree_obs.astype(np.float32)
        return tree_obs


    def get_best_actions(self, obs, num_agents):
        predcition, values = self.model.predict(obs)
        actions = {}
        for i in range(num_agents):
            a_dist = predcition[i]
            actions[i] = np.argmax(a_dist)

        return actions


    def get_actions_and_values(self, obs, num_agents):
        predcition, values = self.model.predict(obs)
        actions = {}
        for i in range(num_agents):
            a_dist = predcition[i]
            a = np.random.choice([0,1,2,3,4], p = a_dist)
            actions[i] = a

        return actions, values


    def get_actions(self, obs, num_agents):
        predcition, values = self.model.predict(obs)
        actions = {}
        for i in range(num_agents):
            a_dist = predcition[i]
            a = np.random.choice([0,1,2,3,4], p = a_dist)
            actions[i] = a

        return actions


    def get_values(self, obs, num_agents):
        return self.model.predict(obs)[1]


    def get_observation_builder(self):
        return CombinedObservation([11,11],3)