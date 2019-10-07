import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model,load_model

import numpy as np
import deliverables.input_params as params
from deliverables.observation import TreeObsForRailEnv
from deliverables.utils.shortest_path import ShortestPathPredictorForRailEnv
#from deliverables.observation import CombinedObservation

class AC_Network():
    def __init__(self, global_model, trainer, create_networks=True, lock=None):
        self.global_model = global_model
        self.trainer = trainer
        self.lock = lock

        if create_networks:
            self.actor_model = self.actor_network()
            self.critic_model = self.critic_network()

    def critic_network(self):
        input_tree = layers.Input(shape=list(params.tree_state_size),dtype=tf.float32)
        hidden = layers.BatchNormalization()(input_tree)
        hidden = layers.Dense(512, activation='relu')(hidden)
        hidden = layers.Dropout(0.1)(hidden)
        hidden = layers.Dense(256, activation='relu')(hidden)
        hidden = layers.Dropout(0.1)(hidden)
        hidden = layers.Dense(32, activation='relu')(hidden)
        hidden = layers.Dense(8, activation='relu')(hidden)
        value = layers.Dense(1)(hidden)

        return Model(
            inputs=input_tree,
            outputs=value)

    def actor_network(self):
        input_tree = layers.Input(shape=list(params.tree_state_size),dtype=tf.float32)
        hidden = layers.BatchNormalization()(input_tree)
        hidden = layers.Dense(512, activation='relu')(hidden)
        hidden = layers.Dropout(0.1)(hidden)
        hidden = layers.Dense(256, activation='relu')(hidden)
        hidden = layers.Dropout(0.1)(hidden)
        hidden = layers.Dense(32, activation='relu')(hidden)
        hidden = layers.Dense(8, activation='relu')(hidden)
        policy = layers.Dense(params.number_of_actions, activation='softmax')(hidden)

        return Model(
            inputs=input_tree,
            outputs=policy)


    def update_from(self, from_model):
        self.lock.acquire()
        actor_weights = from_model.actor_model.get_weights()
        critic_weights = from_model.critic_model.get_weights()
        self.lock.release()

        self.actor_model.set_weights(actor_weights)
        self.critic_model.set_weights(critic_weights)


    def save_model(self, model_path, suffix):
        self.actor_model.save(model_path+'/actor_model_' + suffix + '.h5')
        self.critic_model.save(model_path+'/critic_model_' + suffix + '.h5')
        print('New',suffix,'model saved')


    def load_model(self,model_path, suffix):
        self.actor_model = load_model(model_path+'/actor_model_' + suffix + '.h5')
        self.critic_model = load_model(model_path+'/critic_model_' + suffix + '.h5')


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

        self.lock.acquire()

        # Value loss
        with tf.GradientTape() as t:
            value = self.critic_model(obs)
            v_loss = self.value_loss(target_v,value)

        v_local_vars = self.critic_model.trainable_variables
        gradients_v = t.gradient(v_loss, v_local_vars)
        var_norms_critic = tf.linalg.global_norm(v_local_vars)
        gradients_v, grad_norms_v = tf.clip_by_global_norm(gradients_v, params.gradient_norm_critic)
        global_vars_v = self.global_model.critic_model.trainable_variables
        self.trainer.apply_gradients(zip(gradients_v, global_vars_v))

        # Policy loss
        with tf.GradientTape() as t:
            policy = self.actor_model(obs)
            p_loss, entropy = self.policy_loss(advantages, actions, policy)

        p_local_vars = self.actor_model.trainable_variables
        gradients_p = t.gradient(p_loss, p_local_vars)
        var_norms_actor = tf.linalg.global_norm(p_local_vars)
        gradients_p, grad_norms_p = tf.clip_by_global_norm(gradients_p, params.gradient_norm_actor)
        global_vars_p = self.global_model.actor_model.trainable_variables

        self.trainer.apply_gradients(zip(gradients_p, global_vars_p))
        
        self.lock.release()

        return v_loss, p_loss, entropy, grad_norms_p, grad_norms_v, var_norms_actor, var_norms_critic

    def sigmoid(self,x, derivative=False):
        return x*(1-x) if derivative else 1/(1+np.exp(-x))

    def reshape_obs(self, obs):
        obs = list(obs.values())
        tree_obs = np.asarray(obs)
        tree_obs[tree_obs ==  np.inf] = -1
        tree_obs[tree_obs ==  -np.inf] = -2
        tree_obs += 2
        tree_obs /= 40.0
        tree_obs = tree_obs.astype(np.float32)

        return tree_obs


    def get_best_actions(self, obs, num_agents):
        predcition = self.actor_model.predict(obs)
        actions = {}
        for i in range(num_agents):
            a_dist = predcition[i]
            actions[i] = np.argmax(a_dist)

        return actions


    def get_actions_and_values(self, obs, num_agents):
        predcition = self.actor_model.predict(obs)
        values = self.critic_model.predict(obs)
        actions = {}
        for i in range(num_agents):
            a_dist = predcition[i]
            a = np.random.choice([0,1,2,3,4], p = a_dist)
            actions[i] = a

        return actions, values


    def get_actions(self, obs, num_agents):
        predcition = self.actor_model.predict(obs)
        actions = {}
        for i in range(num_agents):
            a_dist = predcition[i]
            a = np.random.choice([0,1,2,3,4], p = a_dist)
            actions[i] = a

        return actions


    def get_values(self, obs, num_agents):
        return self.critic_model.predict(obs)


    def get_observation_builder(self):
        return TreeObsForRailEnv(3, predictor=ShortestPathPredictorForRailEnv(20))