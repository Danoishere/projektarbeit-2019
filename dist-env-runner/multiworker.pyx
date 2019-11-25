import threading
import multiprocess as mp
import numpy as np
import tensorflow as tf
from ctypes import c_bool
import requests

#from flatland.utils.rendertools import RenderTool, AgentRenderVariant
import scipy.signal
from tensorflow.keras.optimizers import RMSprop

from datetime import datetime
from random import choice,uniform, random, getrandbits, seed
from time import sleep
from time import time
import math
#import msvcrt
import os
cwd = os.getcwd()

from rail_env_wrapper import RailEnvWrapper
from flatland.envs.rail_env import RailEnvActions, RailAgentStatus

import constant as const

class KeyboardInterruptError(Exception): pass

def create_worker(name, should_stop):
    print('Create worker', name)
    worker = Worker(name, should_stop)
    return worker.work()


# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]



class Worker():
    def __init__(self, name, should_stop):
        self.should_stop = should_stop
        self.name = "worker_" + str(name)
        self.number = name        
        self.summary_writer = tf.summary.create_file_writer('tensorboard/train_' + str(name))
        
        print('Create', self.name)

        network_mod = __import__("deliverables.network", fromlist=[''])
        network_class = getattr(network_mod, 'AC_Network')

        curriculum_mod = __import__("deliverables.curriculum", fromlist=[''])
        curriculum_class =  getattr(curriculum_mod, 'Curriculum')
        self.curriculum = curriculum_class()
        # Not only create levels with the current curriculum level but also
        # the levels below
        self.curriculum.randomize_level_generation = True

        self.params = __import__("deliverables.input_params", fromlist=[''])
        self.obs_helper = __import__("deliverables.observation", fromlist=[''])
        
        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_model = network_class(True,const.url, self.number)
        self.env = RailEnvWrapper(self.local_model.get_observation_builder())
        

    def work(self):
        try:
            print ("Starting worker " + str(self.number))
            steps_on_level = 0
            self.episode_rewards = []
            self.episode_lengths = []
            self.episode_success = []
            self.episode_agents_arrived = []

            self.stats = []
            self.episode_mean_values = []
            self.local_model.update_from_global_model()
            self.local_model.update_entropy_factor()
            self.episode_count = 0
            self.curriculum.update_env_to_curriculum_level(self.env)
            use_best_actions = False

            time_start = time()
            '''
            env_renderer = RenderTool(self.env.env, gl="PILSVG",
                          agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                          show_debug=False,
                          screen_height=800,  # Adjust these parameters to fit your resolution
                          screen_width=800) 

            '''
            

            while not bool(self.should_stop.value):    

                # Check with server if there is a new curriculum level available
                if self.episode_count % 50 == 0:
                    self.local_model.update_entropy_factor()
                    old_curriculum_level = self.curriculum.current_level
                    self.curriculum.update_curriculum_level()

                    # Only regenerate env on curriculum level change. Otherwise just reset
                    # Important, because otherwise the player doens't see all levels
                    if self.curriculum.current_level != old_curriculum_level:
                        self.curriculum.update_env_to_curriculum_level(self.env)
                        self.episode_count = 0
                        self.stats = []

                episode_done = False

                # Buffer for obs, action, next obs, reward
                episode_buffer = [[] for i in range(self.env.num_agents)]

                # A way to remember if the agent was already done during the last episode
                done_last_step = {i:False for i in range(self.env.num_agents)}

                # Metrics for tensorboard logging
                episode_values = []
                episode_reward = 0
                episode_step_count = 0
                
                obs, info = self.env.reset()
                all_handles = [i for i in range(len(self.env.env.agents))]
                no_reward = {i:0 for i in range(len(self.env.env.agents))}

                prep_steps = 0
                use_best_actions = bool(getrandbits(1))

                done = {i:False for i in range(len(self.env.env.agents))}
                done['__all__'] = False

                tot_step = 0
                tot_nn = 0
                tot_env_s = 0
                tot_just_obs = 0

                episode_start = time()
                obs_builder = self.env.env.obs_builder
                agent_pos = {}
                cancel_episode = False

                
                # env_renderer.env = self.env.env
                # env_renderer.set_new_rail()
                
                max_steps = self.env.env._max_episode_steps - 5

                while not episode_done and not cancel_episode and episode_step_count < max_steps:
                    agents = self.env.env.agents
                    env_actions, nn_actions, v, relevant_obs = self.local_model.get_agent_actions(self.env, obs, info, use_best_actions)
                    next_obs, rewards, done, info = self.env.step(env_actions)

                    #env_renderer.render_env(show=True)

                    handles = []
                    for agent in agents:
                        if agent.position is not None:
                            handles.append((agent.handle, *agent.position, agent.malfunction_data['malfunction'], agent.wait))

                    agent_pos_key = tuple(handles)
                    if agent_pos_key in agent_pos:
                        agent_pos[agent_pos_key] += 1
                    else:
                        agent_pos[agent_pos_key] = 0

                    max_pos_repeation = max(agent_pos.values())
                    if max_pos_repeation > 10:
                        cancel_episode = True

                    prep_steps = 0
                    obs_builder.prep_steps = prep_steps
                    episode_step_count += 1

                    episode_done = done['__all__']
                    if episode_done == True:
                        next_obs = obs

                    for i in relevant_obs:
                        agent_obs = obs[i]
                        agent_action = nn_actions[i]
                        agent_reward = rewards[i]

                        if not done_last_step[i]:
                            episode_buffer[i].append([
                                agent_obs,
                                agent_action,
                                agent_reward,
                                episode_done,
                                v[i]])
                            
                            episode_values.append(v[i])
                            episode_reward += agent_reward
                
                    obs = next_obs              
                    steps_on_level += 1
                    done_last_step = dict(done)

                num_agents_done = 0
                for i in range(self.env.num_agents):
                    if done[i]:
                        num_agents_done += 1

                # Individual rewards
                for i in range(self.env.num_agents):
                    if len(episode_buffer[i]) > 0:
                        if self.env.env.agents[i].status == RailAgentStatus.READY_TO_DEPART:
                            episode_buffer[i][-1][2] -= 1.0
                            episode_reward -= 1.0
                        if done[i]:
                            # If agents could finish the level, 
                            # set final reward for all agents
                            episode_buffer[i][-1][2] += 1.0
                            episode_reward += 1.0
                        elif cancel_episode:
                            episode_buffer[i][-1][2] -= 1.0
                            episode_reward -= 1.0


                episode_time = time() - episode_start

                avg_episode_time = (time()- time_start)/(self.episode_count + 1)
                agents_arrived = num_agents_done/self.env.num_agents

                self.episode_agents_arrived.append(agents_arrived)
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))
                self.episode_success.append(episode_done)

                v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, num_agents_done)
                self.stats.append([v_l, p_l, e_l, g_n, v_n])

                # Save stats to Tensorboard every 5 episodes
                self.log_in_tensorboard()
                self.episode_count += 1

                if self.episode_count % 50 == 0 and self.episode_count != 0:
                    successrate = np.mean(self.episode_agents_arrived[-25:])
                    requests.post(url=const.url + '/report_success', json={'successrate':successrate})
                
                print('Episode', self.episode_count,'of',self.name,'with',episode_step_count,'steps, reward of',episode_reward,', perc. arrived',agents_arrived, ', mean entropy of', np.mean([l[2] for l in self.stats[-1:]]), ', curriculum level', self.curriculum.current_level, ', using best actions:', use_best_actions,', cancel episode:', cancel_episode, ', time', episode_time, ', Avg. time', avg_episode_time)

            return self.episode_count
    
        except KeyboardInterrupt:
            raise KeyboardInterruptError()


    def train(self, rollout, episode_done):
        all_rollouts = []
        all_rewards = []

        for i in range(self.env.num_agents):
            rewards = [row[2] for row in rollout[i]]
            rewards = discount(rewards, self.params.gamma)
            all_rewards.append(rewards)
            all_rollouts += rollout[i]

        discounted_rewards = np.concatenate(all_rewards)
        
        batch_size = 50
        num_batches = math.ceil(len(all_rollouts)/batch_size)
        for batch in range(num_batches):
            idx_start = batch * batch_size
            idx_end = (batch + 1) * batch_size

            batch_rollout = all_rollouts[idx_start:idx_end]
            batch_rewards = discounted_rewards[idx_start:idx_end]
            
            batch_actions = np.asarray([row[1] for row in batch_rollout]) 
            batch_values = np.asarray([row[4] for row in batch_rollout])
            batch_obs = self.obs_helper.buffer_to_obs_lists(batch_rollout)
            batch_advantages = batch_rewards - batch_values

            v_l,p_l,e_l, g_n, v_n = self.local_model.train(batch_rewards, batch_advantages, batch_actions, batch_obs, episode_done)

        return v_l, p_l, e_l, g_n,  v_n

    def log_in_tensorboard(self):
        if self.episode_count % 5 == 0 and self.episode_count != 0:
            mean_length = np.mean(self.episode_lengths[-100:])
            mean_value = np.mean(self.episode_mean_values[-100:])
            mean_success_rate = np.mean(self.episode_success[-100:])
            mean_agents_arrived = np.mean(self.episode_agents_arrived[-100:])
            mean_reward = np.mean(self.episode_rewards[-100:])

            mean_value_loss = np.mean([l[0] for l in self.stats[-1:]])
            mean_policy_loss = np.mean([l[1] for l in self.stats[-1:]])
            mean_entropy_loss = np.mean([l[2] for l in self.stats[-1:]])
            mean_gradient_norm = np.mean([l[3] for l in self.stats[-1:]])
            mean_variable_norm = np.mean([l[4] for l in self.stats[-1:]])

            with self.summary_writer.as_default():
                episode_count = np.int32(self.episode_count)
                lvl = str(self.curriculum.active_level)
                tf.summary.scalar('Lvl '+ lvl+' - Perf/Reward', mean_reward, step=episode_count)
                tf.summary.scalar('Lvl '+ lvl+' - Perf/Length', mean_length, step=episode_count)
                tf.summary.scalar('Lvl '+ lvl+' - Perf/Value', mean_value, step=episode_count)
                tf.summary.scalar('Lvl '+ lvl+' - Perf/Successrate', mean_success_rate, step=episode_count)
                tf.summary.scalar('Lvl '+ lvl+' - Perf/Perc. agents arrived', mean_agents_arrived, step=episode_count)
                tf.summary.scalar('Lvl '+ lvl+' - Losses/Value Loss', mean_value_loss, step=episode_count)
                tf.summary.scalar('Lvl '+ lvl+' - Losses/Policy Loss', mean_policy_loss, step=episode_count)
                tf.summary.scalar('Lvl '+ lvl+' - Losses/Entropy', mean_entropy_loss, step=episode_count)
                tf.summary.scalar('Lvl '+ lvl+' - Losses/Grad Norm', mean_gradient_norm, step=episode_count)
                tf.summary.scalar('Lvl '+ lvl+' - Losses/Var Norm', mean_variable_norm, step=episode_count)
                self.summary_writer.flush()


    


        