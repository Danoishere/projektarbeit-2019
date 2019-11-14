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
        self.entropy_factor = 0.1

        self.gradient_update_interval = 10
        self.last_gradient_update = 0
        

    def get_network_hash(self):
        summary = StringIO()
        self.model.summary(print_fn=lambda x: summary.write(x + '\n'))
        summary = summary.getvalue().encode('utf-8')
        hasher = hashlib.sha1(summary)
        nw_hash = base64.urlsafe_b64encode(hasher.digest()[:10])
        return str(nw_hash, 'utf-8')


    def build_network(self):
        input_vec = layers.Input(shape=params.tot_obs_size,dtype=tf.float32)

        input_actor_rec = layers.Input(shape=(2,params.recurrent_size),dtype=tf.float32)
        input_critic_rec = layers.Input(shape=(2,params.recurrent_size),dtype=tf.float32)

        actor_out, actor_out_rec = self.create_network(input_vec, input_actor_rec)
        critic_out, critic_out_rec = self.create_network(input_vec, input_critic_rec)

        policy = layers.Dense(params.number_of_actions, activation='softmax')(actor_out)
        value = layers.Dense(1)(critic_out)

        model = Model(
            inputs=[input_vec, input_actor_rec, input_critic_rec],
            outputs=[policy, value, actor_out_rec, critic_out_rec])

        return model


    def create_network(self, input, input_rec):
        hidden = layers.Dense(512, activation='relu')(input)
        hidden = layers.Dense(128, activation='relu')(hidden)
        hidden = layers.Dense(64, activation='relu')(hidden)
        hidden = layers.Reshape((1,64))(hidden)
        hidden, state_h, state_c = layers.LSTM(64, return_state=True, return_sequences=False)(hidden, initial_state=[input_rec[:,0], input_rec[:,1]])
        hidden = layers.Dense(64, activation='relu')(hidden)
        hidden = layers.Dense(8, activation='relu')(hidden)

        return hidden, [state_h, state_c]


    def update_from_global_model(self):
        ''' Updates the local copy of the global model 
        '''
        resp = requests.get(url=self.global_model_url + '/get_global_weights')
        weights_str = resp.content
        weights_str = zlib.decompress(weights_str)
        weights = msgpack.loads(weights_str)
        self.model.set_weights(weights)


    def update_entropy_factor(self):
        ''' Updates the local copy of the global model 
        '''
        resp = requests.get(url=self.global_model_url + '/entropy_factor').json()
        new_entropy_factor = resp['entropy_factor']  * 0.1
        if new_entropy_factor != self.entropy_factor:
            print('New entropy factor aquired:', new_entropy_factor)
            self.entropy_factor = new_entropy_factor

    def save_model(self, model_path, suffix):
        self.model.save(model_path+'/model_' + suffix + '.h5')
        print('New',suffix,'model saved')


    def load_model(self,model_path, suffix):
        self.model = load_model(model_path+'/model_' + suffix + '.h5')


    def value_loss(self, rec_reward, est_reward):
        v_l = 0.5 * tf.reduce_sum(tf.square(rec_reward - tf.reshape(est_reward,[-1])))
        return v_l
    

    def policy_loss(self, advantages, actions, policy):
        actions_onehot = tf.one_hot(actions, params.number_of_actions)
        responsible_outputs = tf.reduce_sum(policy * actions_onehot, [1])
        policy_log = tf.math.log(tf.clip_by_value(policy, 1e-20, 1.0))
        entropy = -tf.reduce_sum(policy * policy_log, axis=1)
        policy_loss = tf.math.log(responsible_outputs  + 1e-20)*advantages
        policy_loss = -tf.reduce_sum(policy_loss + entropy * self.entropy_factor)
        return policy_loss, tf.reduce_mean(entropy)


    def train(self, target_v, advantages, actions,  obs):
        # Value loss
        with tf.GradientTape() as tape:
            policy,value,_,_ = self.model(obs)
            v_loss = self.value_loss(target_v, value)
            p_loss, entropy = self.policy_loss(advantages, actions, policy)
            tot_loss = p_loss + v_loss

        local_vars = self.model.trainable_variables
        gradients_new = tape.gradient(tot_loss, local_vars)
        var_norms = tf.linalg.global_norm(local_vars)
        gradients_new, grad_norms = tf.clip_by_global_norm(gradients_new, params.gradient_norm)

        gradients_str = dill.dumps(gradients_new)
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
        all_vec_obs = []
        all_rec_actor_obs = []
        all_rec_critic_obs = []

        for handle in obs:
            agent_obs = obs[handle]
            vec_obs = agent_obs[0]
            rec_actor_obs = agent_obs[1]
            rec_critic_obs = agent_obs[2]

            all_vec_obs.append(vec_obs)
            all_rec_actor_obs.append(rec_actor_obs)
            all_rec_critic_obs.append(rec_critic_obs)

        return [all_vec_obs, all_rec_actor_obs, all_rec_critic_obs]


    def get_best_actions_and_values(self, obs, env):
        if len(obs) == 0:
            return {},{}

        mapping = {}
        idx = 0
        for handle in obs:
            mapping[handle] = idx
            idx += 1

        obs_list = self.obs_dict_to_lists(obs)
        predcition, values, a_rec_h, a_rec_c, c_rec_h, c_rec_c = self.model.predict_on_batch(obs_list)
        actions = {}
        values_dict = {}

        obs_builder = env.obs_builder

        for handle in obs:
            idx = mapping[handle]
            a_dist = predcition[idx]
            actions[handle] = np.argmax(a_dist)
            values_dict[handle] = values[idx,0]

            obs_builder.actor_rec_state[handle] = [a_rec_h[idx], a_rec_c[idx]]
            obs_builder.critic_rec_state[handle] = [c_rec_h[idx], c_rec_c[idx]]

            env.agents[handle].last_action = a_dist

        return actions, values_dict


    def get_actions_and_values(self, obs, env):
        if len(obs) == 0:
            return {},{}

        mapping = {}
        idx = 0
        for handle in obs:
            mapping[handle] = idx
            idx += 1

        obs_list = self.obs_dict_to_lists(obs)
        predcition, values, a_rec_h, a_rec_c, c_rec_h, c_rec_c = self.model.predict_on_batch(obs_list)
        actions = {}
        values_dict = {}

        obs_builder = env.obs_builder

        for handle in obs:
            idx = mapping[handle]
            a_dist = predcition[idx]
            actions[handle] = np.random.choice([0,1,2,3,4], p = a_dist)
            
            values_dict[handle] = values[idx,0]
            obs_builder.actor_rec_state[handle] = [a_rec_h[idx], a_rec_c[idx]]
            obs_builder.critic_rec_state[handle] = [c_rec_h[idx], c_rec_c[idx]]

            env.agents[handle].last_action = a_dist

        return actions, values_dict


    def get_values(self, obs):
        obs_list = self.obs_dict_to_lists(obs)
        return self.model.predict_on_batch(obs_list)[1]


    def get_observation_builder(self):
        return RailObsBuilder()