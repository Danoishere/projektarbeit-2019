from flatland.envs.observations import GlobalObsForRailEnv, TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
# First of all we import the Flatland rail environment
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
# We also include a renderer because we want to visualize what is going on in the environment
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from flatland.envs.malfunction_generators import malfunction_from_params

import numpy as np
import time

from deliverables.network import AC_Network
import deliverables.observation as obs_helper
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.widgets import Button
from flatland.envs.rail_env import RailEnvActions, RailAgentStatus
import msvcrt

def plot_graph(obs):
    G=nx.DiGraph()
    node_id  = 0
    node_id = draw_branch(obs[0], node_id, G, 0, 0)

    if any(obs[1]):
        node_id = draw_branch(obs[1], node_id, G, 1, 1)
    
    if any(obs[2]):
        node_id = draw_branch(obs[2], node_id, G , 2, 2)
    
    pos=nx.get_node_attributes(G,'pos')
    pos = nx.layout.spring_layout(G,pos=pos,iterations=0)

    nodes = nx.draw_networkx_nodes(G, pos)
    edges = nx.draw_networkx_edges(G, pos, arrowstyle='->',arrowsize=20)
                                
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edge_labels(G, pos)

def draw_branch(start_node, node_id, G, y_pos, source_node):
    root_node = node_id
    
    if start_node is not None:
        dir_fac = 1
        if start_node[0] == 'R':
            dir_fac = -1

    for dir_node in start_node:
        if dir_node is None:
            break

        dir = dir_node[0]
        node = dir_node[1]
        if node is None:
            print('No node here')
            break

        node_pos = node_id - root_node
        G.add_node(node_id, node=node, spec='fastest', pos=(node_pos + source_node, dir_fac*y_pos))
        if node_id > root_node:
            G.add_edge(node_id - 1, node_id, l=node.dist_to_next_branch, d=dir, t=node.dist_min_to_target)
        elif source_node != node_id:
            G.add_edge(source_node, node_id, l=node.dist_to_next_branch, d=dir,t=node.dist_min_to_target)

        node_id += 1
    return node_id
        

def punish_impossible_actions(env, obs, actions, rewards):
    for handle in obs:

        agent = env.agents[handle]
        if agent.position is None:
            if actions[handle] != RailEnvActions.MOVE_FORWARD:
                rewards[handle] -= 0.5
            return

        possible_transitions = env.rail.get_transitions(*agent.position, agent.direction)
        num_transitions = np.count_nonzero(possible_transitions)

        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right], relative to the current orientation
        # If only one transition is possible, the forward branch is aligned with it.
        if num_transitions == 1:
            possible_actions = [0, 1, 0]
        else:
            min_distances = []
            possible_actions = []
            for direction in [(agent.direction + i) % 4 for i in range(-1, 2)]:
                if possible_transitions[direction]:
                    possible_actions.append(1)
                else:
                    possible_actions.append(0)

        # Try left but its prohibited
        if actions[handle] == RailEnvActions.MOVE_LEFT and possible_actions[0] == 0:
            rewards[handle] -= 0.2
            print('left pen')
        if actions[handle] == RailEnvActions.MOVE_FORWARD and possible_actions[1] == 0:
            rewards[handle] -= 0.2
            print('forward pen')
        if actions[handle] == RailEnvActions.MOVE_RIGHT and possible_actions[2] == 0:
            rewards[handle] -= 0.2
            print('right pen')


width = 20  # With of map
height = 20  # Height of map
nr_trains = 5 # Number of trains that have an assigned task in the env
cities_in_map = 2  # Number of cities where agents can start or end
seed = 14  # Random seed
grid_distribution_of_cities = False  # Type of city distribution, if False cities are randomly placed
max_rails_between_cities = 1  # Max number of tracks allowed between cities. This is number of entry point to a city
max_rail_in_cities = 2  # Max number of parallel tracks within a city, representing a realistic trainstation

rail_generator = sparse_rail_generator(max_num_cities=cities_in_map,
                                       seed=seed,
                                       grid_mode=grid_distribution_of_cities,
                                       max_rails_between_cities=max_rails_between_cities,
                                       max_rails_in_city=max_rail_in_cities
                                       )

# The schedule generator can make very basic schedules with a start point, end point and a speed profile for each agent.
# The speed profiles can be adjusted directly as well as shown later on. We start by introducing a statistical
# distribution of speed profiles

# Different agent types (trains) with different speeds.
speed_ration_map = {1.: 0.25,  # Fast passenger train
                    1. / 2.: 0.25,  # Fast freight train
                    1. / 3.: 0.25,  # Slow commuter train
                    1. / 4.: 0.25}  # Slow freight train

# We can now initiate the schedule generator with the given speed profiles

schedule_generator = sparse_schedule_generator(speed_ration_map)

# We can furthermore pass stochastic data to the RailEnv constructor which will allow for stochastic malfunctions
# during an episode.

stochastic_data = {
                   'prop_malfunction': 0.3,  # Percentage of defective agents
                   'malfunction_rate': 30,  # Rate of malfunction occurence
                   'min_duration': 3,  # Minimal duration of malfunction
                   'max_duration': 20  # Max duration of malfunction
                   }

# Custom observation builder without predictor
#observation_builder = GlobalObsForRailEnv()

model = AC_Network()
model.load_model('deliverables/model','checkpoint_lvl_0')

# Custom observation builder with predictor, uncomment line below if you want to try this one
observation_builder = model.get_observation_builder()

# Construct the enviornment with the given observation, generataors, predictors, and stochastic data
env = RailEnv(width=width,
              height=height,
              rail_generator=rail_generator,
              schedule_generator=schedule_generator,
              number_of_agents=nr_trains,
              malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),  # Malfunction data generator
              obs_builder_object=observation_builder,
              remove_agents_at_target=True,  # Removes agents at the end of their journey to make space for others
              random_seed=None)

# Initiate the renderer
env_renderer = RenderTool(env, gl="PILSVG",
                          agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                          show_debug=False,
                          screen_height=800,  # Adjust these parameters to fit your resolution
                          screen_width=800)  # Adjust these parameters to fit your resolution

plt.ion()
plt.show()

succ_best = 1
tries_best = 1
succ_stoch = 1
tries_stoch = 1
use_best = False


def is_agent_on_usable_switch(position, dir):
    ''' a tile is a switch with more than one possible transitions for the
        given direction. '''

    if position is None:
        return False

    transition = env.rail.get_transitions(*position, dir)

    if np.sum(transition) == 1:
        return False
    else:
        return True

def is_agent_on_unusable_switch(position, dir):
    ''' a tile is a switch with more than one possible transitions for the
        given direction. '''

    if position is None:
        return False

    possible_transitions = np.sum(env.rail.get_transitions(*position, dir))
    #print(env.rail.get_transitions(*position, dir))
    for d in range(4):
        dir_transitions = np.sum(env.rail.get_transitions(*position, d))
        if dir_transitions > possible_transitions >= 1:
            #print(env.rail.get_transitions(*position, d))
            return True

    return False

def agent_action_to_env_action(agent, agent_action):
    ''' agent actions: left, right, wait
        env actions: 'do nothing, left, forward, right, brake 
    '''
    if agent.position is None:
        return RailEnvActions.MOVE_FORWARD

    if agent_action == 3:
        return RailEnvActions.MOVE_FORWARD

    if agent_action == 2:
        agent.wait = 5
        if agent.speed_data['speed'] > 0:
            return RailEnvActions.STOP_MOVING
        else:
            return RailEnvActions.DO_NOTHING


    dir = agent.direction
    transition = env.rail.get_transitions(*agent.position, agent.direction)

    can_go_left = False
    can_go_forward = False
    can_go_right = False

    if transition[(3 + dir) % 4] == 1:
        can_go_left = True
    if transition[(0 + dir) % 4] == 1:
        can_go_forward = True
    if transition[(1 + dir) % 4] == 1:
        can_go_right = True

    print('Can go left:', can_go_left)
    print('Can go forward:', can_go_forward)
    print('Can go right:', can_go_right)
    
    if agent_action == 0 and can_go_left:
        return RailEnvActions.MOVE_LEFT
    if agent_action == 1 and can_go_right:
        return RailEnvActions.MOVE_RIGHT

    return RailEnvActions.MOVE_FORWARD


def next_pos(position, direction):
    if position is None:
        return None

    transition = env.rail.get_transitions(*position, direction)
    if np.sum(transition) > 1:
        None

    posy = position[0] - transition[0]  + transition[2]
    posx = position[1] + transition[1] - transition[3]

    return [posy, posx]


while True:
    episode_done = False
    episode_reward = 0
    episode_step_count = 0

    obs, info = env.reset()
    obs_builder = env.obs_builder
    env_renderer.set_new_rail()

    while episode_done == False and episode_step_count < 180:
        agents = env.env.agents
        env_actions, nn_actions, v, relevant_obs = model.get_agent_actions(env, obs, info, True)
        next_obs, rewards, done, info = env.step(env_actions)
        env_renderer.render_env(show=True)

        episode_done = done['__all__']
        if episode_done == True:
            next_obs = obs
        
        obs = next_obs               
        episode_step_count += 1


    if use_best:
        if episode_done:
            succ_best += 1
        tries_best +=1
    else:
        if episode_done:
            succ_stoch += 1
        tries_stoch +=1

    use_best = not use_best

    print('Best - Succ:',succ_best, 'of', tries_best, ', Ratio =',succ_best/tries_best)
    print('Stoch - Succ:',succ_stoch, 'of', tries_stoch, ', Ratio =',succ_stoch/tries_stoch)