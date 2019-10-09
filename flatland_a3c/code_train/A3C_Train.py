#!/usr/bin/env python
# coding: utf-8

# This iPython notebook includes an implementation of the [A3C algorithm](https://arxiv.org/pdf/1602.01783.pdf).
# 
# tensorboard --logdir=deliverables/tensorboard
#
#  ##### Enable autocomplete


import threading
import multiprocess as mp

import numpy as np
import tensorflow as tf
from ctypes import c_bool

import scipy.signal
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model

from datetime import datetime
from random import choice
from time import sleep, time
from rail_env_wrapper import RailEnvWrapper
from code_util.checkpoint import CheckpointManager
from code_train.curriculum import CurriculumManager
from deliverables.network import AC_Network
from code_train.multiworker import create_worker

import code_util.constants as const
import deliverables.input_params as params


def start_train(resume):
    lock = mp.Lock()
    global_model = AC_Network(None,None,True,False,lock)
    num_workers = mp.cpu_count()

    should_stop = mp.Value(c_bool, False)
    # Curriculum-manager manages the generation of the levels
    curr_manager = CurriculumManager(should_stop, 'worker_0')

    # Checkpoint-manager saves model-checkpoints
    ckpt_manager = CheckpointManager(global_model, curr_manager, 'worker_0', save_best_after_min=30, save_ckpt_after_min=100)

    start_episode = 0
    if resume == True:
        print ('Loading Model...')
        start_episode = ckpt_manager.load_checkpoint_model()

    # Initial save for global model
    global_model.save_model(const.model_path,'global')

    while not curr_manager.stop_training:
        worker_processes = []

        # Start process 1 - n, running in other processes
        for w_num in range(1,num_workers):
            #create_worker(w_num, trainer,ckpt_manager,curr_manager,start_episode,lock, should_stop)
            process = mp.Process(target=create_worker, args=(w_num,None,curr_manager,start_episode,lock, should_stop))
            process.start()
            sleep(0.5)
            worker_processes.append(process)

        try:
            # Start process 0
            start_episode = create_worker(0,ckpt_manager,curr_manager,start_episode,lock, should_stop)
        except KeyboardInterrupt:
            print('Key-Interrupt')
            should_stop.value = True
            sleep(2.0)
            raise

        

        for p in worker_processes:
            p.join()

        should_stop.value = False
        if curr_manager.level_switch_activated:
            # Save model after each curriculum level
            run_time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            global_model.save_model(const.model_path_hist,'level_' + str(curr_manager.current_level) + '_' + run_time)
            curr_manager.switch_to_next_level()

    print ("Looks like we're done")

if __name__ == "__main__":
    start_train(False)
