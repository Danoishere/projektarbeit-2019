from flask import Flask, jsonify, send_from_directory,send_file, request, Response
from model_server import Singleton
import json_tricks as json
import numpy as np
import dill
import gzip
import sys
from io import BytesIO
from code_util.locking import ReadWriteLock
import time
import threading

app = Flask(__name__)
lock = threading.RLock()
state = Singleton.get_instance()
network_hash = state.global_model.network_hash

# Save weights so they can be received by the workers
state.global_model.model.save_weights('deliverables/weights.h5')

@app.route('/send_gradient', methods=['POST'])
def post_update_weights():
    state.episode_count += 1
    print('Update Nr. ',state.episode_count)
    gradient_str = request.stream.read()
    gradients = dill.loads(gradient_str)
  
    lock.acquire()
    global_vars = state.global_model.model.trainable_variables
    state.trainer.apply_gradients(zip(gradients, global_vars))
    lock.release()

    return "OK"


@app.route('/get_global_weights')
def get_global_weights():
    return send_from_directory('deliverables','weights.h5')


@app.route('/network_file')
def get_network_file():
    return send_from_directory('deliverables', 'network.py')


@app.route('/config_file')
def get_config_file():
    return send_from_directory('deliverables', 'input_params.py')


@app.route('/observation_file')
def get_observation_file():
    return send_from_directory('deliverables', 'observation.py')


def numpy_to_gzip_response(arrays):
    b = BytesIO()
    np.savez_compressed(b, arr=arrays)
    return send_file(b, mimetype='application/gzip')
