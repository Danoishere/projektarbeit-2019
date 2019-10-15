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
        #input_map = layers.Input(shape=list(params.map_state_size),dtype=tf.float32)
        #input_grid = layers.Input(shape=list(params.grid_state_size),dtype=tf.float32)
        input_vec_tree = layers.Input(shape=list(params.vec_tree_state_size),dtype=tf.float32)

        actor_out = self.create_network(input_vec_tree)
        critic_out = self.create_network(input_vec_tree)

        policy = layers.Dense(params.number_of_actions, activation='softmax')(actor_out)
        value = layers.Dense(1)(critic_out)

        return Model(
            inputs=input_vec_tree,
            outputs=[policy, value])


    def create_network(self, input_vec_tree):
        '''
        map_conv = layers.Conv2D(64,(4,4), activation='relu')(input_map)
        map_conv = layers.Dense(32, activation='relu')(map_conv)
        map_conv = layers.Flatten()(map_conv)
        map_conv = layers.Dense(16, activation='relu')(map_conv)

        grid_conv = layers.Conv2D(32, (3,3), activation='relu')(input_grid)
        grid_dense = layers.Flatten()(grid_conv)
        grid_dense = layers.Dense(64)(grid_dense)
        
        tree_dense = layers.Dense(256, activation='relu')(input_vec_tree)
        tree_dense = layers.Dense(64, activation='relu')(input_vec_tree)

        
        hidden = layers.concatenate([grid_dense, tree_dense, map_conv])
        '''

        hidden = layers.Dense(512, activation='relu')(input_vec_tree)
        hidden = layers.Dense(256, activation='relu')(hidden)
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

        # Send gradient update and receive new global weights
        data_resp = requests.post(
            url=self.global_model_url + '/send_gradient', 
            data=gradients_str)

        self.update_from_global_model()
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
            '.' : 0,
            'F': 0.25,
            'L': 0.5,
            'R': 0.75,
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


    def reshape_obs(self, obs):
        all_obs = []
        for i in range(len(obs)):
            current_node = obs[i]
            if current_node is None:
                obs_agent = np.zeros((params.feature_branches,(params.tree_depth+1)*params.num_features))
                obs_agent = obs_agent.flatten()
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

                all_obs.append(obs_agent)

        return np.vstack(all_obs).astype(np.float32)


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
        return TreeObsForRailEnv(params.tree_depth, ShortestPathPredictorForRailEnv(40))