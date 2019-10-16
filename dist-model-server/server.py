from flask import Flask, jsonify, send_from_directory,send_file, request, Response
from model_server import Singleton
import numpy as np
import dill
import gzip
from io import BytesIO
from code_util.locking import ReadWriteLock
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
    state.ckpt_manager.try_save_model(state.episode_count, 0)
    global_vars = state.global_model.model.trainable_variables
    state.trainer.apply_gradients(zip(gradients, global_vars))
    lock.release()

    return get_global_weights()


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
