from evaluation_parcour import Evaluator
from neuralnetwork import create_model, single_obs_to_tensor, convert_global_obs, obs_list_to_tensor

import constants as const
import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

def start_model_evaluation(model_name = 'model12_03.h5'):
    model = create_model()
    model.load_weights(model_name)
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
    evaluator.start_evaluation(model_name)
    evaluator.analyze_stats(model_name)

    print(evaluator.stats)

if __name__ == '__main__':
    start_model_evaluation()

    