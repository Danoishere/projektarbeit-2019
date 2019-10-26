import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model,load_model

import numpy as np
import requests 
import dill
import deliverables.input_params as params

from io import StringIO
from flatland.envs.observations import TreeObsForRailEnv
from deliverables.observation import RailObsBuilder
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

import base64
import hashlib
import urllib
import msgpack
import zlib


class AC_Network():
    def __init__(self, create_network=True, global_model_url = '', name = '', is_training = False):
        self.name = str(name)
        self.is_training = is_training
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
        input_tree = layers.Input(shape=params.tree_state_size,dtype=tf.float32)
        input_vec = layers.Input(shape=params.vec_state_size,dtype=tf.float32)

        actor_out = self.create_network(input_tree, input_vec)
        critic_out = self.create_network(input_tree, input_vec)

        policy = layers.Dense(params.number_of_actions, activation='softmax')(actor_out)
        value = layers.Dense(1)(critic_out)

        model = Model(
            inputs=[input_tree, input_vec],
            outputs=[policy, value])

        return model


    def create_network(self, input_tree, input_vec):
        conv = layers.Reshape((params.tree_state_size,1))(input_tree)
        conv = layers.Conv1D(filters = 40, kernel_size =(params.num_features), strides=(params.num_features), activation='relu')(conv)
        conv = layers.Flatten()(conv)
        conv = layers.Dense(256, activation='relu')(conv)
        conv = layers.Dense(64, activation='relu')(conv)

        hidden = layers.concatenate([conv, input_vec])
        hidden = layers.Dense(64, activation='relu')(hidden)
        hidden = layers.Dense(8, activation='relu')(hidden)

        return hidden


    def update_from_global_model(self):
        ''' Updates the local copy of the global model 
        '''
        resp = requests.get(url=self.global_model_url + '/get_global_weights')
        weights_str = resp.content
        weights_str = zlib.decompress(weights_str)
        weights = msgpack.loads(weights_str)
        self.model.set_weights(weights)


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
        entropy = -(policy * policy_log) * params.entropy_factor
        policy_loss = -tf.reduce_sum(tf.math.log(responsible_outputs  + 1e-10)*advantages) - entropy
        return policy_loss, tf.reduce_mean(entropy)


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
        gradients_str = zlib.compress(gradients_str)

        # Send gradient update and receive new global weights
        resp = requests.post(
            url=self.global_model_url + '/send_gradient', 
            data=gradients_str)

        weights_str = resp.content
        weights_str = zlib.decompress(weights_str)
        weights = msgpack.loads(weights_str)

        self.model.set_weights(weights)
        return v_loss, p_loss, entropy, grad_norms, var_norms


    def get_best_actions(self, obs):
        obs_list = self.obs_dict_to_lists(obs)
        predcition, _ = self.model.predict_on_batch(obs_list)
        actions = {}
        for i in obs:
            a_dist = predcition[i]
            actions[i] = np.argmax(a_dist)

        return actions


    def obs_dict_to_lists(self, obs):
        all_tree_obs = []
        all_vec_obs = []
        for handle in obs:
            agent_obs = obs[handle]
            tree_obs = agent_obs[0]
            vec_obs = agent_obs[1]
            all_tree_obs.append(tree_obs)
            all_vec_obs.append(vec_obs)

        return [all_tree_obs, all_vec_obs]


    def get_best_actions_and_values(self, obs):
        obs_list = self.obs_dict_to_lists(obs)
        predcition, values = self.model.predict_on_batch(obs_list)
        actions = {}
        values_dict = {}
        for handle in obs:
            a_dist = predcition[handle]
            actions[handle] = np.argmax(a_dist)
            values_dict[handle] = values[handle,0]

        return actions, values_dict


    def get_actions_and_values(self, obs):
        obs_list = self.obs_dict_to_lists(obs)
        predcition, values = self.model.predict_on_batch(obs_list)
        actions = {}
        values_dict = {}
        for handle in obs:
            a_dist = predcition[handle]
            a = np.random.choice([0,1,2,3,4], p = a_dist)
            actions[handle] = a
            values_dict[handle] = values[handle,0]

        return actions, values_dict


    def get_actions(self, obs):
        obs_list = self.obs_dict_to_lists(obs)
        predcition, _ = self.model.predict_on_batch(obs_list)
        actions = {}
        for handle in obs:
            a_dist = predcition[handle]
            a = np.random.choice([0,1,2,3,4], p = a_dist)
            actions[handle] = a

        return actions


    def get_values(self, obs):
        obs_list = self.obs_dict_to_lists(obs)
        return self.model.predict_on_batch(obs_list)[1]


    def get_observation_builder(self):
        return RailObsBuilder()