

import os
def delete_files(dir_name, extensions):
    test = os.listdir(dir_name)
    for item in test:
        for ext in extensions:
            if item.endswith(ext):
                os.remove(os.path.join(dir_name, item))
                print('Delete file', os.path.join(dir_name, item))

delete_files('./deliverables', ['.c', '.pyd', '.so','.o'])
delete_files('.', ['.c', '.pyd', '.so','.o'])

myCmd = 'python setup.py build_ext --inplace'
os.system(myCmd)

from flask import Flask, jsonify, send_from_directory,send_file, request, Response
from flatland.envs.observations import TreeObsForRailEnv

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


@app.route('/send_gradient', methods=['POST'])
def post_update_weights():

    state.episode_count += 1
    print('Update Nr. ',state.episode_count)
    gradient_str = request.stream.read()
    gradient_str = zlib.decompress(gradient_str)
    gradients = dill.loads(gradient_str)

    lock.acquire()
    global_vars = state.global_model.model.trainable_variables
    state.trainer.apply_gradients(zip(gradients, global_vars))
    lock.release()

    return get_global_weights()

@app.route('/report_success', methods=['POST'])
def post_success():
    data = request.get_json()
    state.ckpt_manager.try_save_model(state.episode_count, data['successrate'])
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
    weights = state.global_model.model.get_weights()
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
        entropy_factor = 0.0025
    elif state.curriculum_level == 1:
        entropy_factor = 0.0025
    elif state.curriculum_level == 2:
        entropy_factor = 0.0025
    elif state.curriculum_level == 3:
        entropy_factor = 0.0025
    elif state.curriculum_level == 4:
        entropy_factor = 0.0025
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
