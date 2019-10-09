import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model,load_model

import numpy as np
import deliverables.input_params as params
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from deliverables.observation import CombinedObservation

class AC_Network():
    def __init__(self, trainer, model_path, create_networks=True, load_global_model=False, lock=None):
        self.trainer = trainer
        self.lock = lock

        if create_networks:
            self.actor_model = self.actor_network()
            self.critic_model = self.critic_network()
        
        if load_global_model:
            lock.acquire()
            self.global_actor = load_model(model_path + '/actor_model_global.h5')
            self.global_critic = load_model(model_path + '/critic_model_global.h5')
            lock.release()

    def critic_network(self):
        input_map = layers.Input(shape=list(params.map_state_size),dtype=tf.float32)
        input_grid = layers.Input(shape=list(params.grid_state_size),dtype=tf.float32)
        input_vec_tree = layers.Input(shape=list(params.vec_tree_state_size),dtype=tf.float32)

        map_dense = layers.Flatten()(input_map)
        map_dense = layers.Dense(128, activation='relu')(map_dense)
        map_dense = layers.Dense(64, activation='relu')(map_dense)

        grid_conv = layers.Conv2D(62, (3,3))(input_grid)
        grid_dense = layers.Flatten()(grid_conv)
        grid_dense = layers.Dense(128)(grid_dense)
        
        tree_dense = layers.Dense(256, activation='relu')(input_vec_tree)
        tree_dense = layers.Dense(128, activation='relu')(input_vec_tree)

        hidden = layers.concatenate([map_dense, grid_dense, tree_dense])
        hidden = layers.Dense(256, activation='relu')(hidden)
        hidden = layers.Dropout(0.1)(hidden)
        hidden = layers.Dense(32, activation='relu')(hidden)
        hidden = layers.Dense(8, activation='relu')(hidden)
        value = layers.Dense(1)(hidden)

        return Model(
            inputs=[input_map, input_grid, input_vec_tree],
            outputs=value)

    def actor_network(self):
        input_map = layers.Input(shape=list(params.map_state_size),dtype=tf.float32)
        input_grid = layers.Input(shape=list(params.grid_state_size),dtype=tf.float32)
        input_vec_tree = layers.Input(shape=list(params.vec_tree_state_size),dtype=tf.float32)

        map_dense = layers.Flatten()(input_map)
        map_dense = layers.Dense(128, activation='relu')(map_dense)
        map_dense = layers.Dense(64, activation='relu')(map_dense)

        grid_conv = layers.Conv2D(62, (3,3))(input_grid)
        grid_dense = layers.Flatten()(grid_conv)
        grid_dense = layers.Dense(128)(grid_dense)
        
        tree_dense = layers.Dense(256, activation='relu')(input_vec_tree)
        tree_dense = layers.Dense(128, activation='relu')(input_vec_tree)

        hidden = layers.concatenate([map_dense, grid_dense, tree_dense])
        hidden = layers.Dense(256, activation='relu')(hidden)
        hidden = layers.Dropout(0.1)(hidden)
        hidden = layers.Dense(32, activation='relu')(hidden)
        policy = layers.Dense(params.number_of_actions, activation='softmax')(hidden)

        return Model(
            inputs=[input_map, input_grid, input_vec_tree],
            outputs=policy)


    def update_from_global_model(self, model_path):
        self.lock.acquire()
        self.actor_model.load_weights(model_path + '/actor_model_global.h5')
        self.critic_model.load_weights(model_path + '/critic_model_global.h5')
        self.lock.release()


    def update_global_model(self, model_path):
        self.global_actor.load_weights(model_path + '/actor_model_global.h5')
        self.global_critic.load_weights(model_path + '/critic_model_global.h5')

    def save_global_model(self, model_path):
        self.global_actor.save(model_path + '/actor_model_global.h5')
        self.global_critic.save(model_path + '/critic_model_global.h5')
        

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


    def train(self, target_v, advantages, actions, obs, model_path):

        # Value loss
        with tf.GradientTape() as t_v:
            value = self.critic_model(obs)
            v_loss = self.value_loss(target_v,value)

        # Policy loss
        with tf.GradientTape() as t_p:
            policy = self.actor_model(obs)
            p_loss, entropy = self.policy_loss(advantages, actions, policy)

        v_local_vars = self.critic_model.trainable_variables
        gradients_v = t_v.gradient(v_loss, v_local_vars)
        var_norms_critic = tf.linalg.global_norm(v_local_vars)
        gradients_v, grad_norms_v = tf.clip_by_global_norm(gradients_v, params.gradient_norm_critic)

        p_local_vars = self.actor_model.trainable_variables
        gradients_p = t_p.gradient(p_loss, p_local_vars)
        var_norms_actor = tf.linalg.global_norm(p_local_vars)
        gradients_p, grad_norms_p = tf.clip_by_global_norm(gradients_p, params.gradient_norm_actor)
        
        self.lock.acquire()
        self.update_global_model(model_path)
        global_vars_v = self.global_critic.trainable_variables
        global_vars_p = self.global_actor.trainable_variables
        self.trainer.apply_gradients(zip(gradients_v, global_vars_v))
        self.trainer.apply_gradients(zip(gradients_p, global_vars_p))
        self.save_global_model(model_path)

        self.lock.release()

        return v_loss, p_loss, entropy, grad_norms_p, grad_norms_v, var_norms_actor, var_norms_critic

    def sigmoid(self,x, derivative=False):
        return x*(1-x) if derivative else 1/(1+np.exp(-x))

    def reshape_obs(self, obs):
        tree_obs = np.asarray(obs['obs1'])

        #len_obs = tree_obs.shape[1]
        #len_cell = 11

        #num_cells = int(len_obs/len_cell)

        tree_obs[tree_obs ==  np.inf] = -1
        tree_obs[tree_obs ==  -np.inf] = -2
        #tree_obs = np.asarray(np.split(tree_obs, num_cells, axis=1))
        #tree_obs = np.swapaxes(tree_obs, 0,1)
        tree_obs += 2
        tree_obs /= 40.0

        '''
        dist_to_target = tree_obs[:,:,0]
        dist_to_other_agents_target = tree_obs[:,:,1]
        dist_to_other_agent = tree_obs[:,:,2]
        possible_conflicts = tree_obs[:,:,3]
        unusable_switch = tree_obs[:,:,4]
        branch_length = tree_obs[:,:,5]
        dist_to_target_with_this_path = tree_obs[:,:,6]
        num_agents_same_dir = tree_obs[:,:,7]
        num_agents_opposite_dir = tree_obs[:,:,8]
        blocked_for_n_more_steps = tree_obs[:,:,9]
        slowest_agent = tree_obs[:,:,10]

        tree_obs = tree_obs*0.5
        tree_obs = tree_obs - 5
        tree_obs = self.sigmoid(tree_obs)
        '''

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
        return CombinedObservation([11,11],3)