from deliverables.observation import CombinedObservation
from tensorflow.keras import models
from code_benchmark.benchmark import Evaluator
from deliverables.network import AC_Network
import code_util.constants as const 
import numpy as np

def run_benchmark():
    model = AC_Network(None,None,False)
    model.load_model(const.model_path, const.suffix_best)
    
    evaluator = Evaluator(model.get_actions, model.get_observation_builder())
    evaluator.start_evaluation()
    evaluator.save_stats_to_csv('benchmark')

if __name__ == "__main__":
    run_benchmark()




