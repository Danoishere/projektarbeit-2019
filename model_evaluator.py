from evaluation_parcour import Evaluator
from neuralnetwork import create_model, single_obs_to_tensor, convert_global_obs, obs_list_to_tensor

import constants as const
import tensorflow as tf
import numpy as np

def start_model_evaluation(model_name = 'model_16_42'):
    model = create_model()
    model.load_weights('model16_42.h5')
    def model_function(observations,num_agents):
        actions = {}
        current_observations = convert_global_obs(observations)
        for i in range(num_agents):
            current_observation = current_observations[i]
            logits, _ = model(single_obs_to_tensor(current_observation))
            probs = tf.nn.softmax(logits).numpy()[0]
            actions[i] = np.random.choice(const.ACTIONS, p=probs)

        return actions

    evaluator = Evaluator()
    evaluator.set_model(model_function)
    evaluator.start_evaluation()

    print(evaluator.stats)

if __name__ == '__main__':
    start_model_evaluation()

    