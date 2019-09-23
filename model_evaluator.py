from evaluation_parcour import Evaluator
from neuralnetwork import create_model, single_obs_to_tensor, convert_global_obs, obs_list_to_tensor

import constants as const
import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

def start_model_evaluation(model_name = 'model00_22'):
    model = create_model()
    model.load_weights(model_name + '.h5')
    def model_function(observations,num_agents):
        current_observations = convert_global_obs(observations)
        current_observations = obs_list_to_tensor(current_observations)
        logits, _ = model(current_observations)
        probs = tf.nn.softmax(logits).numpy()

        actions = {}
        for i in range(num_agents):
            actions[i] = np.random.choice(const.ACTIONS, p=probs[i])

        return actions

    evaluator = Evaluator()
    evaluator.set_model(model_function)
    evaluator.start_evaluation()
    evaluator.analyze_stats(model_name)

    print(evaluator.stats)

def start_random_evaluation(report_name = 'random'):
    def model_function(observations,num_agents):
        actions = {}
        for i in range(num_agents):
            actions[i] = np.random.choice(const.ACTIONS)

        return actions

    evaluator = Evaluator()
    evaluator.set_model(model_function)
    evaluator.start_evaluation()
    evaluator.analyze_stats(report_name)

    print(evaluator.stats)

if __name__ == '__main__':
    #start_random_evaluation()
    start_model_evaluation()

    