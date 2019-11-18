#!/usr/bin/env python
# coding: utf-8

# This iPython notebook includes an implementation of the [A3C algorithm](https://arxiv.org/pdf/1602.01783.pdf).
# 
# tensorboard --logdir=deliverables/tensorboard
#
#  ##### Enable autocomplete

import multiprocess as mp
import numpy as np
import tensorflow as tf
from ctypes import c_bool
import requests

import os
myCmd = 'python setup.py build_ext --inplace'
os.system(myCmd)

# import shared directory
import os, sys; 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)) + os.sep + 'shared')

from time import sleep, time
import constant as const
import urllib

mp.set_start_method('spawn', True)

def start_train(resume):

    dir_name = "deliverables"
    test = os.listdir(dir_name)

    for item in test:
        if item.endswith(".pyd"):
            os.remove(os.path.join(dir_name, item))
        if item.endswith(".c"):
            os.remove(os.path.join(dir_name, item))

    urllib.request.urlretrieve(const.url + '/file/network.pyx', 'deliverables/network.pyx')
    urllib.request.urlretrieve(const.url + '/file/input_params.py', 'deliverables/input_params.py')
    urllib.request.urlretrieve(const.url + '/file/observation.pyx', 'deliverables/observation.pyx')
    urllib.request.urlretrieve(const.url + '/file/curriculum.py', 'deliverables/curriculum.py')

    myCmd = 'python setup_deliverables.py build_ext --inplace'
    os.system(myCmd)

    params = __import__("deliverables.input_params", fromlist=[''])

    # Wait with this import until we compiled all required modules!
    from multiworker import create_worker

    num_workers = mp.cpu_count() - 1
    should_stop = mp.Value(c_bool, False)

    resp = requests.get(url=const.url + '/round').json()
    round = resp['round']


    while True:
        worker_processes = []
        print('----------------------')
        print('Round ', round)
        print('----------------------')
        #create_worker(0, round, should_stop, round*params.ev_episodes)

        
        # Start process 1 - n, running in other processes
        for w_num in range(0,num_workers):
            process = mp.Process(target=create_worker, args=(w_num, round, should_stop, round*params.ev_episodes))
            process.start()
            sleep(0.5)
            worker_processes.append(process)
        try:
            for p in worker_processes:
                p.join()
        except KeyboardInterrupt:
            should_stop.value = True
        

        resp = requests.post(url=const.url + '/finish_round')
        round += 1

    print ("Looks like we're done")

if __name__ == "__main__":
    start_train(False)
