from deliverables.network import AC_Network
from code_train.curriculum import CurriculumManager
from code_util.checkpoint import CheckpointManager

from tensorflow.keras.optimizers import RMSprop
from datetime import datetime

import deliverables.input_params as params
import code_util.constants as const 
import subprocess
import json


class Singleton:
    # Here will be the instance stored.
    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if Singleton.__instance == None:
            Singleton()
        return Singleton.__instance 

    def __init__(self):
        """ Virtually private constructor. """
        if Singleton.__instance != None:
            raise Exception("This class is a singleton!")

        Singleton.__instance = self
        init_global_model(self)
        

def init_global_model(singelton):
    # TODO: Refactor to arguments

    singelton.resume = False
    singelton.global_model = AC_Network()
    singelton.trainer = RMSprop(learning_rate=params.learning_rate)

    # Curriculum-manager manages the generation of the levels
    singelton.curr_manager = CurriculumManager()

    # Checkpoint-manager saves model-checkpoints
    singelton.ckpt_manager = CheckpointManager(
        singelton.global_model, 
        singelton.curr_manager, 
        save_best_after_min=30, 
        save_ckpt_after_min=100)

    singelton.episode_count  = 0
    if singelton.resume == True:
        print ('Loading model...')
        singelton.episode_count = singelton.ckpt_manager.load_checkpoint_model()
    else:
        print('Create new run...')
        save_new_run_info()

    # Initial save for global model
    singelton.global_model.save_model(const.model_path, 'global')




def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])

def save_new_run_info():
    run_start_time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    github_hash = get_git_revision_hash()
    benchmark_score = -1

    run_info = {
            'run_start_time':run_start_time,
            'description':'TODO',
            'github_hash': github_hash.decode("utf-8"),
            'benchmark_score':benchmark_score
        }

    with open(const.run_info_path + 'run_info.json', 'w') as json_file:
        json.dump(run_info, json_file)