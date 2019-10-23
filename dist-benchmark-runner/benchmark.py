#!/usr/bin/env python
# coding: utf-8

# This iPython notebook includes an implementation of the [A3C algorithm](https://arxiv.org/pdf/1602.01783.pdf).
# 
# tensorboard --logdir=deliverables/tensorboard
#
#  ##### Enable autocomplete

# import shared directory
import os, sys; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)) + os.sep + 'shared')

import multiprocess as mp
import numpy as np
import tensorflow as tf
from ctypes import c_bool
from time import sleep, time
from runstatistics import RunStatistics
from rail_env_wrapper import RailEnvWrapper

import pandas as pd
import constant as const
import urllib
import requests
import json
import dill

mp.set_start_method('spawn', True)
SEED = 345123


class Benchmark:
    def __init__(self):
        urllib.request.urlretrieve(const.url + '/network_file', 'deliverables/network.py')
        urllib.request.urlretrieve(const.url + '/config_file', 'deliverables/input_params.py')
        urllib.request.urlretrieve(const.url + '/observation_file', 'deliverables/observation.py')
        urllib.request.urlretrieve(const.url + '/curriculum_file', 'deliverables/curriculum.py')

        curriculum_mod = __import__("deliverables.curriculum", fromlist=[''])
        curriculum_class =  getattr(curriculum_mod, 'Curriculum')
        self.curriculum = curriculum_class()
        self.curriculum.seed = SEED

        network_mod = __import__("deliverables.network", fromlist=[''])
        network_class = getattr(network_mod, 'AC_Network')

        self.obs_helper = __import__("deliverables.observation", fromlist=[''])
        self.model = network_class(True, const.url)
        self.env = RailEnvWrapper(self.model.get_observation_builder())


    def start_benchmark(self):
        while True:
            num_success = 0
            episode_count = 0
            self.stats = []
            self.model.update_from_global_model()
            self.trained_on_curr_level = self.get_curriculum_lvl()
                       
            print('New benchmark run with updated model.')
            print('Round 1 - simple environment with one agent')
            self.change_grid_round1()
            self.run_episodes(1, 1)

            '''
            print('Round 2 - Two agents, larger environment')
            self.change_grid_round2()
            self.run_episodes(2, 20)

            print('Round 3 - Three agents, but similar environment')
            self.change_grid_round3()
            self.run_episodes(3, 20)

            print('Round 4 - Four agents, same size')
            self.change_grid_round4()
            self.run_episodes(4, 20)
            
            print('Round 5 - Large environment, 10 agents')
            self.change_grid_round5()
            self.run_episodes(5, 20)
            '''

            self.submit_benchmark_report()



    def run_to_end(self):
        episode_done = False
        episode_buffer = [[] for i in range(self.env.num_agents)]
        done_last_step = {i:False for i in range(self.env.num_agents)}
        episode_reward = 0
        episode_step_count = 0
        
        obs, info = self.env.reset()

        while episode_done == False and episode_step_count < self.env.max_steps:
            actions = self.model.get_best_actions(obs)
            next_obs, rewards, done, info = self.env.step(actions)

            episode_done = done['__all__']
            if episode_done == True:
                next_obs = obs

            for i in range(self.env.num_agents):
                agent_obs = obs[i] 
                agent_action = actions[i]
                agent_reward = rewards[i]
                agent_next_obs =  next_obs[i] 

                if not done_last_step[i]:
                    episode_buffer[i].append([
                        agent_obs,
                        agent_action,
                        agent_reward,
                        agent_next_obs,
                        episode_done,
                        0])
                    
                    episode_reward += agent_reward
            
            obs = next_obs               
            episode_step_count += 1
            done_last_step = dict(done)

        num_agents_done = self.count_agents_done(done, self.env.num_agents)
        return num_agents_done, episode_reward, episode_step_count


    def save_stats(self, num_agents_done, total_reward, steps_needed, round):
        run_stats = RunStatistics(
                        self.env.num_agents,
                        num_agents_done,
                        total_reward,
                        self.env.width,
                        self.env.height,
                        steps_needed,
                        round,
                        self.trained_on_curr_level
                    )

        print(run_stats)
        self.stats.append(run_stats)


    def change_grid_round1(self):
        self.curriculum.current_level = 0
        self.curriculum.update_env_to_curriculum_level(self.env)


    def change_grid_round2(self):
        self.curriculum.current_level = 1
        self.curriculum.update_env_to_curriculum_level(self.env)
        

    def change_grid_round3(self):
        self.curriculum.current_level = 2
        self.curriculum.update_env_to_curriculum_level(self.env)


    def change_grid_round4(self):
        self.curriculum.current_level = 3
        self.curriculum.update_env_to_curriculum_level(self.env)


    def change_grid_round5(self):
        self.curriculum.current_level = 4
        self.curriculum.update_env_to_curriculum_level(self.env)


    def run_episodes(self, round_nr, num_episodes):
        for r in range(num_episodes):
            num_agents_done, total_reward, steps_needed = self.run_to_end()
            self.save_stats(num_agents_done, total_reward, steps_needed, round_nr)

    def submit_benchmark_report(self):
        df = pd.DataFrame.from_records([s.to_dict() for s in self.stats])
        report_str = dill.dumps(df)
        resp = requests.post(
            url=const.url + '/send_benchmark_report', 
            data=report_str)
        print('Benchmark report submitted')
        


    def save_stats_to_csv(self,run_name):
        df = pd.DataFrame.from_records([s.to_dict() for s in self.stats])
        df.to_csv(const.benchmark_path + 'benchmark_report.csv')


    def update_run_info_benchmark_score(self):
        df = pd.DataFrame.from_records([s.to_dict() for s in self.stats])
        benchmark_score = df['num_agents_done'].sum()/df['num_agents'].sum()
        with open(const.run_info_path + 'run_info.json', 'rw') as json_file:
            run_info = json.load(json_file)
            run_info['benchmark_score'] = benchmark_score
            json.dump(run_info, json_file)


    def sum_up_dict(self,dict, num_agents):
        val = 0
        for i in range(num_agents):
            val += dict[i]
        return val


    def count_agents_done(self,done_dict, num_agents):
        agents_done = 0
        for i in range(num_agents):
            if done_dict[i] == True:
                agents_done += 1
        
        return agents_done


    def get_curriculum_lvl(self):
        data = requests.get(url=const.url + '/curriculum_level').json()
        return data['curriculum_lvl']
        
    
if __name__ == "__main__":
    benchmark = Benchmark()
    benchmark.start_benchmark()
