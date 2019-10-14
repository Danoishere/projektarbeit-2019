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
from multiworker import create_worker

import constant as const
import requests
import urllib

mp.set_start_method('spawn', True)

def start_train(resume):
    
    #urllib.request.urlretrieve(const.url + '/network_file', 'deliverables/network.py')
    urllib.request.urlretrieve(const.url + '/config_file', 'deliverables/input_params.py')
    urllib.request.urlretrieve(const.url + '/observation_file', 'deliverables/observation.py')

    num_workers = mp.cpu_count() - 2
    lock = mp.Lock()
    should_stop = mp.Value(c_bool, False)

    while True:
        worker_processes = []

        create_worker(0, should_stop)

        # Start process 1 - n, running in other processes
        for w_num in range(1,num_workers):
            process = mp.Process(target=create_worker, args=(w_num, should_stop))
            process.start()
            sleep(0.5)
            worker_processes.append(process)

        try:
            for p in worker_processes:
                p.join()
        except KeyboardInterrupt:
            should_stop.value = True

    print ("Looks like we're done")

if __name__ == "__main__":
    start_train(False)
