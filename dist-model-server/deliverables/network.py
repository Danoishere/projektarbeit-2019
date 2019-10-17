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
        input_vec_tree = layers.Input(shape=params.vec_tree_state_size,dtype=tf.float32)

        actor_out = self.create_network(input_vec_tree)
        critic_out = self.create_network(input_vec_tree)

        policy = layers.Dense(params.number_of_actions, activation='softmax')(actor_out)
        value = layers.Dense(1)(critic_out)

        model = Model(
            inputs=input_vec_tree,
            outputs=[policy, value])
        #model.summary()
        return model


    def create_network(self, input_vec_tree):
        conv = layers.Reshape((params.vec_tree_state_size,1))(input_vec_tree)
        conv = layers.Conv1D(kernel_size =(params.num_features), strides=(params.num_features),  filters = 20, activation='relu')(conv)
        conv = layers.Flatten()(conv)
        conv = layers.Dense(300, activation='relu')(conv)
        hidden = layers.Dense(64, activation='relu')(conv)
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
        entropy = - tf.reduce_sum(policy * policy_log) * params.entropy_factor
        policy_loss = -tf.reduce_mean(tf.math.log(responsible_outputs  + 1e-10)*advantages) - entropy 
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


    def sigmoid(self,x, derivative=False):
        return x*(1-x) if derivative else 1/(1+np.exp(-x))

    def get_shortest_way_from(self, entry_dir, start_node):
        selected_nodes = []
        selected_nodes.append((entry_dir,start_node)) # Root

        shortest_way_idx = 'INIT'
        shortest_way = 1000
        found_target = False

        current_node = start_node
        while shortest_way_idx != 'NA' and not found_target and current_node is not None:
            shortest_way_idx = 'NA'
            
            for k in current_node.childs:
                child = current_node.childs[k]
                if child != -np.inf:
                    if child.dist_own_target_encountered != 0 and child.dist_own_target_encountered < 1000:
                        found_target = True
                        shortest_way_idx = k
                    elif child.dist_min_to_target < shortest_way and not found_target:
                        shortest_way = child.dist_min_to_target
                        shortest_way_idx = k

            if shortest_way_idx != 'NA':
                next_node = current_node.childs[shortest_way_idx]
                selected_nodes.append((shortest_way_idx, next_node))
                current_node = next_node

        return selected_nodes

    def get_ordered_children(self, node):
        #if node is None:
        #    return []

        children = []
        for k in node.childs:
            child = node.childs[k]
            if child != -np.inf:
                children.append((k,child))
        
        children = sorted(children, key=lambda t: np.min([t[1].dist_min_to_target, t[1].dist_own_target_encountered]))
        return children

    def normalize_field(self, field, norm_val=100):
        if field == np.inf or field == -np.inf:
            return 0
        else:
            return 0.1 + field/norm_val
        
        

    def node_to_obs(self, node_tuple):

        if node_tuple is None:
            return [0]*params.num_features

        dir = node_tuple[0]
        node = node_tuple[1]

        dir_dict = {
            '.' : 0.1,
            'F': 0.4,
            'L': 0.6,
            'R': 0.7,
        }

        dir_num = dir_dict[dir]
        obs = [
            dir_num,
            self.normalize_field(node.dist_min_to_target),
            self.normalize_field(node.dist_other_agent_encountered),
            self.normalize_field(node.dist_other_target_encountered),
            self.normalize_field(node.dist_own_target_encountered),
            self.normalize_field(node.dist_potential_conflict),
            self.normalize_field(node.dist_to_next_branch),
            self.normalize_field(node.dist_unusable_switch),
            self.normalize_field(node.num_agents_malfunctioning,10),
            self.normalize_field(node.num_agents_opposite_direction, 10),
            self.normalize_field(node.num_agents_ready_to_depart, 20),
            self.normalize_field(node.num_agents_same_direction, 20),
            node.speed_min_fractional
        ]

        return obs


    def reshape_obs(self, obs, info):
        all_obs = []
        for i in range(len(obs)):
            current_node = obs[i]
            if current_node is None:
                obs_agent = np.zeros((params.feature_branches,(params.tree_depth+1)*params.num_features))
                obs_agent = obs_agent.flatten()
                obs_agent = np.insert(obs_agent,0,[0]*params.num_features)
                all_obs.append(obs_agent)
            else:

                # Fastest way from root
                fastest_way = self.get_shortest_way_from('.',current_node)

                sorted_children = self.get_ordered_children(current_node)

                alt_way_1 = [None]* params.tree_depth
                # Try to take second best solution at next intersection
                if len(sorted_children) > 1:
                    alt_node_1 = sorted_children[1]
                    alt_way_1 = self.get_shortest_way_from(alt_node_1[0], alt_node_1[1])

                alt_way_2 = [None]*params.tree_depth
                # Try to take third best solution at next intersection
                if len(sorted_children) > 2:
                    alt_node_2 = sorted_children[2]
                    alt_way_2 = self.get_shortest_way_from(alt_node_2[0], alt_node_2[1])
                
                alt_way_3 = [None]*(params.tree_depth-1)
                alt_way_4 = [None]*(params.tree_depth-1)
                # Try to take second best solution at second next intersection
                if len(fastest_way) > 1:
                    sorted_children = self.get_ordered_children(fastest_way[1][1])
                    if len(sorted_children) > 1:
                        alt_node_3 = sorted_children[1]
                        alt_way_3 = self.get_shortest_way_from(alt_node_3[0], alt_node_3[1])

                    if len(sorted_children) > 2:
                        alt_node_4 = sorted_children[2]
                        alt_way_4 = self.get_shortest_way_from(alt_node_4[0], alt_node_4[1])

                # Fill missing nodes to tree-depth-length
                for j in range(len(fastest_way), params.tree_depth+1):
                    fastest_way.append(None)

                for n in fastest_way:
                    self.node_to_obs(n)

                obs_layers = [fastest_way, alt_way_1, alt_way_2, alt_way_3, alt_way_4]

                obs_agent = np.zeros((params.feature_branches,(params.tree_depth+1)*params.num_features))
                for layer_idx in range(len(obs_layers)):
                    layer = obs_layers[layer_idx]
                    for node_idx in range(len(layer)):
                        node = layer[node_idx]
                        node_obs = self.node_to_obs(node)
                        obs_agent[layer_idx,node_idx*params.num_features:node_idx*params.num_features + params.num_features] = node_obs


                obs_agent = obs_agent.flatten()

                # Insert additional vector for later obs
                obs_agent = np.insert(obs_agent,0,[0]*params.num_features)
                if info['action_required'][i]:
                    obs_agent[0] = 1.0
                if info['malfunction'][i] == 1:
                    obs_agent[1] = 1.0

                obs_agent[2] =info['speed'][i]
                obs_agent[3] =info['status'][i].value

                all_obs.append(obs_agent)

        return np.vstack(all_obs).astype(np.float32)


    def get_best_actions(self, obs, num_agents):
        predcition, _ = self.model.predict_on_batch(obs)
        actions = {}
        for i in range(num_agents):
            a_dist = predcition[i]
            actions[i] = np.argmax(a_dist)

        return actions

    def get_best_actions_and_values(self, obs, num_agents):
        predcition, values = self.model.predict_on_batch(obs)
        actions = {}
        for i in range(num_agents):
            a_dist = predcition[i]
            actions[i] = np.argmax(a_dist)

        return actions, values


    def get_actions_and_values(self, obs, num_agents):
        predcition, values = self.model.predict_on_batch(obs)
        actions = {}
        for i in range(num_agents):
            a_dist = predcition[i]
            a = np.random.choice([0,1,2,3,4], p = a_dist)
            actions[i] = a

        return actions, values


    def get_actions(self, obs, num_agents):
        predcition, values = self.model.predict_on_batch(obs)
        actions = {}
        for i in range(num_agents):
            a_dist = predcition[i]
            a = np.random.choice([0,1,2,3,4], p = a_dist)
            actions[i] = a

        return actions


    def get_values(self, obs, num_agents):
        return self.model.predict_on_batch(obs)[1]


    def get_observation_builder(self):
        return TreeObsForRailEnv(params.tree_depth, ShortestPathPredictorForRailEnv(40))