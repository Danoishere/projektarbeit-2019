import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import random
import threading
import multiprocessing
import numpy as np
import argparse
import matplotlib.pyplot as plt
import time
import tensorflow as tf

from datetime import datetime
from queue import Queue
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D,concatenate

from observation import RawObservation
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool
from flatland.envs.observations import TreeObsForRailEnv, LocalObsForRailEnv, GlobalObsForRailEnv
from flatland.envs.generators import complex_rail_generator, 
from flatland.core.grid.grid4_astar import a_star

tf.enable_eager_execution()

parser = argparse.ArgumentParser(description='Run A3C algorithm on the game '
                                                                                         'Cartpole.')
parser.add_argument('--algorithm', default='a3c', type=str,
                                        help='Choose between \'a3c\' and \'random\'.')
parser.add_argument('--train', dest='train', action='store_true',
                                        help='Train our model.')
parser.add_argument('--lr', default=0.001,
                                        help='Learning rate for the shared optimizer.')
parser.add_argument('--update-freq', default=20, type=int,
                                        help='How often to update the global model.')
parser.add_argument('--max-eps', default=100000, type=int,
                                        help='Global maximum number of episodes to run.')
parser.add_argument('--gamma', default=0.99,
                                        help='Discount factor of rewards.')
parser.add_argument('--save-dir', default='/tmp/', type=str,
                                        help='Directory in which you desire to save the model.')
args = parser.parse_args()

NUMBER_OF_AGENTS = 3
ACTIONS = [0,1,2,3,4]
ACTION_SIZE = len(ACTIONS)
SHOULD_RENDER = False
SAVE_INTERVAL_EPS = 50
STATE_SIZE = (6,20,20)
MAX_EPISODES_PLAY = 20


simplicity_start = 0


def update_rail_gen(env):
    global simplicity_start
    simplicity_start += 0.001
    current_difficulty = int(np.round(simplicity_start))
    env.rail_generator = complex_rail_generator( nr_start_goal=np.max([current_difficulty,NUMBER_OF_AGENTS]),
                                        nr_extra=current_difficulty,
                                        min_dist=5,
                                        max_dist=99999,
                                        seed=random.randint(0,100000))                                  

def create_env():
    env = RailEnv(
                width=20,
                height=20,
                rail_generator = complex_rail_generator( nr_start_goal=6,
                                        nr_extra=4,
                                        min_dist=5,
                                        max_dist=99999,
                                        seed=random.randint(0,100000)),
                number_of_agents=NUMBER_OF_AGENTS,
                obs_builder_object=RawObservation([20,20]))
    env.invalid_action_penalty = -10
    return env

def convert_global_obs(global_obs):
    observations = []
    for i in range(NUMBER_OF_AGENTS):
        agent_obs = global_obs[i]
        state1 = np.array(agent_obs[0])
        state2 = np.array(agent_obs[1])
        observation1 = state1.astype(np.float).reshape((1,state1.shape[0],state1.shape[1],state1.shape[2]))
        observation2 = state2.astype(np.float).reshape((1,state2.shape[0]))
        observations.append([observation1, observation2])

    return observations

def single_obs_to_tensor(observation):
    obs1 = tf.convert_to_tensor(observation[0], dtype=tf.float32)
    obs2 = tf.convert_to_tensor(observation[1], dtype=tf.float32)
    return [obs1,obs2]

def obs_list_to_tensor(observations):
    t_obs1 = []
    t_obs2 = []
    for i in range(len(observations)):
        observation = observations[i]
        t_obs1.append(observation[0][0])
        t_obs2.append(observation[1][0])

    t_obs1 = tf.convert_to_tensor(t_obs1, dtype=tf.float32)
    t_obs2 = tf.convert_to_tensor(t_obs2, dtype=tf.float32)
    return [t_obs1,t_obs2]


def create_model():
    batch_size=21
    i1 = Input(shape=(6,20,20),batch_size=batch_size)
    c = Conv2D(20, kernel_size=(1,5))(i1)
    c = MaxPooling2D()(c)
    c = Conv2D(20, kernel_size=(1,2))(c)
    c = MaxPooling2D()(c)
    f1 = Flatten()(c)

    res_i1 = Flatten()(i1)

    i2 = Input(shape=(6,),batch_size=batch_size)
    r = Dense(64)(i2)
    r = Dense(64)(r)
    r = Flatten()(r)
    i = concatenate([f1,r])
    
    # Value network
    v = Dense(300, activation='relu')(i)
    v = Dense(300, activation='relu')(v)
    v = Dense(300, activation='relu')(v)
    v = concatenate([v,res_i1,i2])
    v = Dense(300, activation='relu')(v)
    v = Dense(200, activation='relu')(v)
    v = Dense(200, activation='relu')(v)
    v = Dense(200, activation='relu')(v)
    v = Dense(200, activation='relu')(v)
    v = Dense(100, activation='relu')(v)
    value = Dense(1)(v)

    c = Conv2D(20, kernel_size=(1,5))(i1)
    c = MaxPooling2D()(c)
    c = Conv2D(20, kernel_size=(1,2))(c)
    c = MaxPooling2D()(c)
    f1 = Flatten()(c)

    r = Dense(64)(i2)
    r = Dense(64)(r)
    r = Flatten()(r)
    i = concatenate([f1,r])

    # Policy network
    p = Dense(300, activation='relu')(i)
    p = Dense(300, activation='relu')(p)
    p = Dense(300, activation='relu')(p)
    p = concatenate([p,res_i1,i2])
    p = Dense(300, activation='relu')(p)
    p = Dense(200, activation='relu')(p)
    p = Dense(200, activation='relu')(p)
    p = Dense(200, activation='relu')(p)
    p = Dense(200, activation='relu')(p)
    p = Dense(100, activation='relu')(p)
    policy = Dense(ACTION_SIZE)(p)

    model = Model(inputs=[i1,i2], outputs=[policy,value])
    return model

def record(episode,
                     episode_reward,
                     worker_idx,
                     global_ep_reward,
                     result_queue,
                     total_loss,
                     num_steps):
    """Helper function to store score and print statistics.

    Arguments:
        episode: Current episode
        episode_reward: Reward accumulated over the current episode
        worker_idx: Which thread (worker)
        global_ep_reward: The moving average of the global reward
        result_queue: Queue storing the moving average of the scores
        total_loss: The total loss accumualted over the current episode
        num_steps: The number of steps the episode took to complete
    """
    if global_ep_reward == 0:
        global_ep_reward = episode_reward
    else:
        global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
    print(
            f"Episode: {episode} | "
            f"Moving Average Reward: {int(global_ep_reward)} | "
            f"Episode Reward: {int(episode_reward)} | "
            f"Loss: {int(total_loss / float(num_steps) * 1000) / 1000} | "
            f"Steps: {num_steps} | "
            f"Worker: {worker_idx}"
    )
    result_queue.put(global_ep_reward)
    return global_ep_reward


class RandomAgent:
    """Random Agent that will play the specified game
        Arguments:
            env_name: Name of the environment to be played
            max_eps: Maximum number of episodes to run agent for.
    """
    def __init__(self, env_name, max_eps):
        self.env = create_env()
        self.max_episodes = max_eps
        self.global_moving_average_reward = 0
        self.res_queue = Queue()

    def run(self):
        reward_avg = 0
        for episode in range(self.max_episodes):
            done = False
            self.env.reset()
            reward_sum = 0.0
            steps = 0
            while not done:
                # Sample randomly from the action space and step
                _, reward, done, _ = self.env.step(self.env.action_space.sample())
                steps += 1
                reward_sum += reward
            # Record statistics
            self.global_moving_average_reward = record(episode,
                                                    reward_sum,
                                                    0,
                                                    self.global_moving_average_reward,
                                                    self.res_queue, 0, steps)

            reward_avg += reward_sum
        final_avg = reward_avg / float(self.max_episodes)
        print("Average score across {} episodes: {}".format(self.max_episodes, final_avg))
        return final_avg


class MasterAgent():
    def __init__(self):
        env = create_env()
        self.opt = tf.compat.v1.train.RMSPropOptimizer(0.0001, use_locking=True, decay = 0.99, epsilon = 0.1)
        self.global_model = create_model()
        self.global_model.load_weights('model12_04.h5')

        obs1 = tf.convert_to_tensor(np.random.random((1,6,20,20)), dtype=tf.float32)
        obs2 = tf.convert_to_tensor(np.random.random((1,6)), dtype=tf.float32)

        self.global_model([obs1,obs2])

    def train(self):
        if args.algorithm == 'random':
            random_agent = RandomAgent(self.game_name, args.max_eps)
            random_agent.run()
            return

        res_queue = Queue()
        if SHOULD_RENDER:
            w = Worker(STATE_SIZE,
                                ACTION_SIZE,
                                self.global_model,
                                self.opt, res_queue,
                                0,
                                #save_dir=self.save_dir) for i in range(multiprocessing.cpu_count())]
                            )
            w.run()

        else:
            workers = [Worker(STATE_SIZE,
                                ACTION_SIZE,
                                self.global_model,
                                self.opt, res_queue,
                                i) for i in range(multiprocessing.cpu_count())]
                            #) for i in range(1)]
            
            for i, worker in enumerate(workers):
                print("Starting worker {}".format(i))
                worker.start()
            
            moving_average_rewards = []    # record episode reward to plot
            while True:
                reward = res_queue.get()
                if reward is not None:
                    moving_average_rewards.append(reward)
                else:
                    break
            [w.join() for w in workers]
        
        #plt.plot(moving_average_rewards)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.savefig(os.path.join(self.save_dir, '{} Moving Average.png'.format(self.game_name)))
        plt.show()

    def play(self):
        self.local_model = self.global_model
        env_done = False
        ep_steps = 0
        reward_sum = 0
        ep_num = 0

        self.env = create_env()
        self.env.reset()
        
        env_renderer = RenderTool(self.env)

        while True:
            current_observations = convert_global_obs(self.env.reset())
            env_renderer.set_new_rail()
            pos = {}
            while not env_done and ep_steps < 200:
                print(ep_steps)
                actions = {}
                for i in range(NUMBER_OF_AGENTS):
                    current_observation = current_observations[i]
                    t_obs = single_obs_to_tensor(current_observation)
                    logits, _ = self.local_model(t_obs)
                    probs = tf.nn.softmax(logits).numpy()[0]
                    actions[i] = np.random.choice(ACTIONS, p=probs)
                    print('Agent', i ,'does action', actions[i] )

                current_observations, rewards, done, _ = self.env.step(actions)
                current_observations = convert_global_obs(current_observations)
                env_done = done['__all__']
                env_renderer.render_env(show=True, frames=False, show_observations=True)

                # End run if trains are stuck for more than 4 steps in same position
                agents_state = tuple([i.position for i in self.env.agents])
                if agents_state in pos.keys():
                    pos[agents_state] += 1
                else:
                    pos[agents_state] = 1
    
                is_stuck = False
                for state in pos:
                    if pos[state] > 4:
                        is_stuck = True
                        env_done = True
                        break


                ep_steps += 1

            ep_steps = 0
            env_done = False

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []


class Worker(threading.Thread):
    # Set up global variables across different threads
    global_episode = 0
    # Moving average reward
    global_moving_average_reward = 0
    best_score = -10000
    save_lock = threading.Lock()

    def __init__(self,
                    state_size,
                    action_size,
                    global_model,
                    opt,
                    result_queue,
                    idx):

        super(Worker, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.result_queue = result_queue
        self.global_model = global_model
        self.opt = opt
        self.local_model = create_model()
        self.worker_idx = idx
        self.env = create_env()
        self.ep_loss = 0.0

    def run(self):
        total_step = 1
        mems = []
        dist = []
        for i in range(NUMBER_OF_AGENTS):
            mems.append(Memory())
            dist.append(1000)

        if SHOULD_RENDER:
            env_renderer = RenderTool(self.env)
        
        while Worker.global_episode < args.max_eps:
            update_rail_gen(self.env)
            current_observations = convert_global_obs(self.env.reset())
            if SHOULD_RENDER:
                env_renderer.set_new_rail()

            ep_reward = 0.
            ep_steps = 0
            ep_loss = 0

            update_counter = 0
            env_done = False
            pos = {}

            for i in range(NUMBER_OF_AGENTS):
                dist[i] = 1000

            while not env_done and ep_steps < 200:
                actions = {}
                for i in range(NUMBER_OF_AGENTS):
                    current_observation = current_observations[i]
                    logits, _ = self.local_model(single_obs_to_tensor(current_observation))
                    probs = tf.nn.softmax(logits).numpy()[0]
                    actions[i] = np.random.choice(ACTIONS, p=probs)
                    print('Agent', i ,'does action', actions[i] )

                next_observations, rewards, done, _ = self.env.step(actions)
                next_observations = convert_global_obs(next_observations)

                env_done = done['__all__']

                # End run if trains are stuck for more than 4 steps in same position
                agents_state = tuple([i.position for i in self.env.agents])
                if agents_state in pos.keys():
                    pos[agents_state] += 1
                else:
                    pos[agents_state] = 1
    
                is_stuck = False
                for state in pos:
                    if pos[state] > 7:
                        is_stuck = True
                        env_done = True
                        break
                
                for i in range(NUMBER_OF_AGENTS):
                    agent = self.env.agents[i]
                    grid = np.zeros((20,20),dtype=np.uint16)
                    path_to_target = a_star(self.env.rail.transitions, grid, agent.position, agent.target)
                    current_path_length = len(path_to_target)
                    last_path_length = dist[i]
                    if current_path_length < last_path_length:
                        rewards[i] += 0.2

                    dist[i] = current_path_length
                
                if SHOULD_RENDER:
                    env_renderer.render_env(show=True, frames=False, show_observations=True)

                ep_reward += np.sum(np.fromiter(rewards.values(), dtype=float))

                for i in range(NUMBER_OF_AGENTS):
                    obs = current_observations[i]
                    action = actions[i]
                    reward = rewards[i]
                    agent_done = done[i]
                    agent_memory = mems[i]
                    agent_memory.store(obs, action, reward)
                    
                # if env_done or ep_steps > 199:
                if update_counter == args.update_freq or env_done or ep_steps > 199:
                    # Calculate gradient wrt to local model. We do so by tracking the
                    # variables involved in computing the loss by using tf.GradientTape
                    for i in range(NUMBER_OF_AGENTS):
                        next_observation = next_observations[i]
                        agent_done = done[i]
                        agent_memory = mems[i]
                        with tf.GradientTape() as tape:
                            total_loss = self.compute_loss(agent_done,
                                                            next_observation,
                                                            agent_memory,
                                                            args.gamma)
                        ep_loss += total_loss
                        print('Loss:', ep_loss)

                        # Calculate local gradients
                        grads = tape.gradient(total_loss, self.local_model.trainable_weights)
                        # Push local gradients to global model
                        self.opt.apply_gradients(zip(grads, self.global_model.trainable_weights))
                        # Update local model with new weights
                        self.local_model.set_weights(self.global_model.get_weights())

                        agent_memory.clear()
                        update_counter = 0

                    if env_done and not is_stuck:    # done and print information
                        Worker.global_moving_average_reward = \
                            record(Worker.global_episode, ep_reward, self.worker_idx,
                                        Worker.global_moving_average_reward, self.result_queue,
                                        ep_loss, ep_steps)

                        # We must use a lock to save our model and to print to prevent data races.
                        #if total_step % SAVE_INTERVAL_EPS == 0:
                        with Worker.save_lock:
                            print(f'Saving model with: {ep_reward}')
                            current_time = datetime.now().strftime('%H_%M')
                            #self.global_model.save_weights('model_'+ current_time +'_'+str(ep_reward)+'_'+str(simplicity_start)+'.h5')
                            self.global_model.save_weights('model'+ current_time +'.h5')
                            Worker.best_score = ep_reward
                        Worker.global_episode += 1
                        
                ep_steps += 1
                update_counter += 1
                current_observations = next_observations
                total_step += 1
            
                
        self.result_queue.put(None)

    def compute_loss(self,done,last_state,memory,gamma=0.99):
        if done:
            # terminal
            reward_sum = 0.    
        else:
            # Episode not done yet. Use Value-Function for approximated remainding reward
            last_state = single_obs_to_tensor(last_state)
            _, value = self.local_model(last_state)
            reward_sum = value.numpy()[0]

        # Get discounted rewards
        discounted_rewards = []
        for reward in memory.rewards[::-1]:    # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)

        discounted_rewards.reverse()
        #states = np.vstack(memory.states)
        # For all visited states get policy/value
        states = obs_list_to_tensor(memory.states)
        logits, values = self.local_model(states)
        
        # Advantage = reward - value (expected reward)
        advantage = np.array(discounted_rewards) - values
        # Value loss
        value_loss = advantage ** 2

        # Calculate our policy loss
        policy = tf.nn.softmax(logits)
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=policy, logits=logits)

        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.actions, logits=logits)
        policy_loss *= tf.stop_gradient(advantage)
        policy_loss -= 0.3 * entropy
        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))

        return total_loss


if __name__ == '__main__':
    print(args)
    master = MasterAgent()
    if args.train:
        master.train()
    else:
        master.play()
