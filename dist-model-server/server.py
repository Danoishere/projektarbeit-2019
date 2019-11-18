import os
dir_name = "deliverables"
test = os.listdir(dir_name)

for item in test:
    if item.endswith(".pyd"):
        os.remove(os.path.join(dir_name, item))
    if item.endswith(".c"):
        os.remove(os.path.join(dir_name, item))


from flask import Flask, jsonify, send_from_directory,send_file, request, Response

myCmd = 'python setup.py build_ext --inplace'
os.system(myCmd)

from flatland.envs.observations import TreeObsForRailEnv
from deliverables.network import AC_Network

import code_util.constants as const


from model_server import Singleton
import tensorflow as tf
import numpy as np
import pandas as pd
import dill
import gzip
from io import BytesIO
from datetime import datetime
import time
import threading
import json
import msgpack
import zlib



app = Flask(__name__)
lock = threading.RLock()
state = Singleton.get_instance()
network_hash = state.global_model.network_hash

model_dict = {}
update_cnt = 0
round = 0
model = AC_Network()


@app.route('/send_model/<float:successrate>', methods=['POST'])
def send_model(successrate):
    global update_cnt

    weights_str = request.stream.read()
    weights = msgpack.loads(weights_str)

    lock.acquire()
    model_dict[update_cnt] = {
        'successrate' : successrate,
        'weights' : weights
    }

    update_cnt += 1

    lock.release()

    return 'OK'

@app.route('/finish_round', methods=['POST'])
def finish_round():
    global model_dict
    global round

    best_rate = 0
    best_key = None
    for k in model_dict:
        if model_dict[k]['successrate'] >= best_rate:
            best_rate = model_dict[k]['successrate']
            best_key = k
    
    if best_key is None:
        raise ValueError('Something went wrong. No models submitted')

    print('Select model', k, ' with successrate of ', best_rate)
    model.model.set_weights(model_dict[best_key]['weights'])
    model_dict = {}
    round += 1

    model.model.save_model(const.model_path, const.suffix_best +'_round_'+ round)

    return 'OK'



@app.route('/send_benchmark_report', methods=['POST'])
def post_send_benchmark_report():
    report_time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    report_str = request.stream.read()
    report = dill.loads(report_str)
    report.to_csv('./deliverables/benchmark_report_' + report_time + '.csv')
    return "OK"


@app.route('/get_global_weights')
def get_global_weights():
    lock.acquire()
    weights = model.model.get_weights()
    lock.release()
    weights_str = msgpack.dumps(weights)
    weights_str = zlib.compress(weights_str)
    return weights_str


@app.route('/increase_curriculum_level')
def increase_curriculum_level():
    state.curriculum_level += 1


@app.route('/curriculum_level')
def get_curriculum_level():
    data = { 'curriculum_lvl' : state.curriculum_level }
    return jsonify(data)


@app.route('/entropy_factor')
def get_entropy_factor():
    entropy_factor = 0.0

    if state.curriculum_level == 0:
        entropy_factor = 0.003
    elif state.curriculum_level == 1:
        entropy_factor = 0.001
    elif state.curriculum_level == 2:
        entropy_factor = 0.0005
    elif state.curriculum_level == 3:
        entropy_factor = 0.000001
    elif state.curriculum_level == 4:
        entropy_factor = 0.0000001
    else:
        entropy_factor = 0.0

    '''
    elif state.episode_count < 30000:
        entropy_factor = 0.005
    elif state.episode_count < 50000:
        entropy_factor = 0.001
    elif state.episode_count < 100000:
        entropy_factor = 0.0001
    '''

    data = { 'entropy_factor' : entropy_factor }
    return jsonify(data)


@app.route('/curriculum_file')
def get_curriculum_file():
    return send_from_directory('deliverables', 'curriculum.py')


@app.route('/network_file')
def get_network_file():
    return send_from_directory('deliverables', 'network.py')


@app.route('/config_file')
def get_config_file():
    return send_from_directory('deliverables', 'input_params.py')


@app.route('/observation_file')
def get_observation_file():
    return send_from_directory('deliverables', 'observation.py')

@app.route('/file/<string:filename>')
def get_file(filename):
    return send_from_directory('deliverables', filename)
