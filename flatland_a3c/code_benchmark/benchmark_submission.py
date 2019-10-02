from observation import CombinedObservation
from tensorflow.keras import models
from code_benchmark.benchmark import Evaluator
import numpy as np

submission_id = 'example_submission'

# Todo: Replace by onedrive
model_path = 'submissions/' + submission_id
model = models.load_model(model_path + '/best_model.h5')

def make_prediction(obs,num_agents):
    predcition = model.predict([obs[0],obs[1],obs[2],obs[3]])
    actions = {}
    for i in range(num_agents):
        a_dist = predcition[0][i]
        a = np.random.choice([0,1,2,3,4], p = a_dist)
        actions[i] = a

    return actions

submission = {
    'observation': CombinedObservation([11,11],2),
    'get_policy': make_prediction
}

def run_benchmark():
    evaluator = Evaluator()
    evaluator.set_benchmark_submission(submission)
    evaluator.start_evaluation()
    evaluator.save_stats_to_csv('benchmark')

if __name__ == "__main__":
    run_benchmark()




