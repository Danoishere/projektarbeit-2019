from flatland.envs.observations import GlobalObsForRailEnv, TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
# First of all we import the Flatland rail environment
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
# We also include a renderer because we want to visualize what is going on in the environment
from flatland.utils.rendertools import RenderTool, AgentRenderVariant

import numpy as np
import time

from deliverables.network import AC_Network
import deliverables.observation as obs_helper
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.widgets import Button

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
        


width = 30  # With of map
height = 30  # Height of map
nr_trains = 1 # Number of trains that have an assigned task in the env
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
model.load_model('deliverables/model','global')

# Custom observation builder with predictor, uncomment line below if you want to try this one
observation_builder = model.get_observation_builder()

# Construct the enviornment with the given observation, generataors, predictors, and stochastic data
env = RailEnv(width=width,
              height=height,
              rail_generator=rail_generator,
              schedule_generator=schedule_generator,
              number_of_agents=nr_trains,
              stochastic_data=stochastic_data,  # Malfunction data generator
              obs_builder_object=observation_builder,
              remove_agents_at_target=True  # Removes agents at the end of their journey to make space for others
              )

# Initiate the renderer
env_renderer = RenderTool(env, gl="PILSVG",
                          agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                          show_debug=False,
                          screen_height=800,  # Adjust these parameters to fit your resolution
                          screen_width=800)  # Adjust these parameters to fit your resolution

plt.ion()
plt.show()


while True:
    episode_done = False
    episode_reward = 0
    episode_step_count = 0

    obs, info = env.reset()
    #plot_graph(obs_helper.graph_list)

    env_renderer.set_new_rail()

    while episode_done == False and episode_step_count < 200:
        # Usually, this part is handled in the network, but to get
        # the probabilities, we do it ourselfs
        obs = model.obs_dict_to_lists(obs)
        predcition, _ = model.model.predict_on_batch(obs)
        actions = {}

        for i in range(nr_trains):
            a_dist = predcition[i]
            actions[i] = np.random.choice([0,1,2,3,4], p=a_dist) #np.argmax(a_dist)

        plt.clf()
        
        plt.subplot(2,1,1)
        plt.bar([0,1,2,3,4],predcition[0],tick_label=['do nothing', 'forward', 'left', 'right', 'stop'])
        
        next_obs, rewards, done, info = env.step(actions)
        #plt.subplot(2,1,2)
        
        #plot_graph(obs_helper.graph_list)

        #plt.xlim([-1.1,1.1])
        #plt.ylim([-1.4,1.4])

        #plt.waitforbuttonpress()
        plt.draw()
    
        env_renderer.render_env(show=True)

        episode_done = done['__all__']
        if episode_done == True:
            next_obs = obs
        
        obs = next_obs               
        episode_step_count += 1