"""
Author: S.Huschauer
Date: 31.08.2019
Email: huschste@students.zhaw.ch
Associated with ZHAW datalab
This code is not exhaustive!
"""

import numpy as np
from model import FlatNet
from loader import FlatData
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
import queue
import random
import copy
import multiprocessing 
from loss import FlatLoss
import time

class FlatLink():
    """
    Linker for environment.
    Training: Generate training data, accumulate n-step reward and train model.
    Play: Execute trained model.
    """
    def __init__(self,size_x=20,size_y=10):
        self.n_step_return_ = 100
        self.gamma_ = 0.99
        self.batch_size = 16000
        self.num_epochs = 20
        self.play_epochs = 100
        self.a = 3.57
        self.alpha_rv = stats.alpha(self.a)
        self.alpha_rv.random_state = np.random.RandomState(seed=342423)
        self.time_stack  = 5
        self.queue = None
        self.max_iter =  self.n_step_return_
        self.dtype = torch.float
        self.play_game_number = 10
        self.number_of_levels = 10
        self.size_x=size_x
        self.size_y=size_y
        self.replay_buffer_size = 4000
        self.buffer_size = 16000
        self.best_loss = 10000000000.0
        self.normalize_reward = 100.0
        self.weights = None
        self.model = None
        self.gpu_ = -1
        self.buffer_threshold = -1.0
        self.epsilon_start  = 2.0
        self.epsilon_end = 0.15
        self.epsilon = 0.0
        self.one_ = 1
        self.state = None

    def init_model(self):
        self.model = FlatNet(5*self.time_stack)
        self.set_gpu()

    def set_gpu(self):
        if self.gpu_>-1:
            self.cuda = True
            self.device = torch.device('cuda', self.gpu_)
            self.model.cuda(self.device)
        else:
            self.cuda = False
            self.device = torch.device('cpu', 0)

    def load_(self, file='model_a3c_i.pt'):
        self.model = torch.load(file)
        print(self.model)

    def load_weights(self, file = 'model_a3c_weight.tar'):
        #state_dict = torch.load(file)
        self.model.load_state_dict( state_dict)


    def load_cpu(self, file):

        mod = torch.load(file)
        print(mod)
        state_dict = mod.cpu().state_dict()
  
        self.model.load_state_dict( state_dict)
        try:
            torch.save(self.model, 'model_a3c_cpu.pt')
                
        except:
            print("Model is not saved!!!")
    
    def test_random_move(self):
        move =  np.argmax(self.alpha_rv.rvs(size=5))
        print("move ",move)
   
    def get_action(self, policy):
        """
        Return either predicted (from policy) or random action
        """
        if np.abs(stats.norm(1).rvs()) < self.epsilon:
            move =  np.argmax(self.alpha_rv.rvs(size=len(policy)))
            return move, 0
        else:
            xk = np.arange(len(policy))
            custm = stats.rv_discrete(name='custm', values=(xk, policy))
            val = custm.rvs()
            return val, 1

    def accumulate(self, memory,state_value, done_):

        """
        memory has form (state, action, reward, after_state, done)
        Accumulate b-step reward and put to training-queue
        """
        n = min(len(memory), self.n_step_return_)
        curr_state = memory[0][0]
        action =memory[0][1]
       
        n_step_return = 0
        for i in range(n):
            reward  = memory[i][2]
            n_step_return += np.power(self.gamma_,i+1) * reward

        state_value = memory[n-1][5]
        done = memory[n-1][4]
      
        n_step_return -= curr_state[1][2]/self.size_x/self.size_y
        if not done:
            
            n_step_return += np.power(self.gamma_,n)*state_value[0]
        else:
            n_step_return += 5.0

        if done_:
            n_step_return += 10.0
            
        if memory!=[]:
            memory.pop(0)
      
        n_step_return /= self.normalize_reward
        n_step_return = np.clip(n_step_return, -1., 1.)
        return curr_state, action, n_step_return , memory, done

   
    def training(self,observations,targets,rewards):
        """
        Run model training on accumulated training experience
        """
        self.model.train()
        data_train = FlatData(observations,targets,rewards,self.model.num_actions_,self.device, self.dtype)
        dataset_sizes = len(data_train)
        dataloader = torch.utils.data.DataLoader(data_train, batch_size=self.batch_size, shuffle=True,num_workers=0)
        best_model_wts = None
        #optimizer= optim.Adam(self.model.parameters(),lr=0.001,weight_decay=0.02,amsgrad=True)
        optimizer= optim.RMSprop(self.model.parameters(),lr=1e-4)
        best_loss  = 10000000000.0
        loss_builder = FlatLoss()
        # TODO early stop
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for observation, obs_2, target, reward in dataloader:
                observation = observation.to(device = self.device, dtype=self.dtype)
                obs_2 = obs_2.to(device = self.device, dtype=self.dtype)
                target = target.to(device = self.device, dtype=self.dtype)
                reward = reward.to(device = self.device, dtype=self.dtype)
                with torch.set_grad_enabled(True):

                    optimizer.zero_grad()
                    policy, value = self.model.forward(observation,obs_2)
                    loss = loss_builder.loss(policy,target,value,reward,obs_2)
                    loss.backward()
                    optimizer.step()
                    running_loss += torch.abs(loss).item() * observation.size(0)
               
            epoch_loss = running_loss / dataset_sizes
           
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                self.weights = best_model_wts
                try:
                    torch.save(self.model, 'model_a3c_i.pt')
                    torch.save(self.weights, 'model_a3c_weights.pt')
                except:
                    print("Model is not saved!!!")
        print('Last Loss {}, Best Loss {}'.format(epoch_loss, best_loss))
        if best_model_wts is not None:
            self.model.load_state_dict(best_model_wts)


    def predict(self, observation):
        """
        Forward pass on model
        Returns: Policy, Value
        """
        state = observation[0]
        state2 = observation[1]
        observation1 = state.astype(np.float).reshape((1,state.shape[0],state.shape[1],state.shape[2]))
        observation2 = state2.astype(np.float).reshape((1,state2.shape[0]))
        val1 = torch.from_numpy(observation1.astype(np.float)).to(device = self.device, dtype=self.dtype)
        val2 = torch.from_numpy(observation2.astype(np.float)).to(device = self.device, dtype=self.dtype)
        p, v = self.model.forward(val1,val2)

        return p.cpu().detach().numpy().reshape(-1), v.cpu().detach().numpy().reshape(-1)

    
    def trainer(self,buffer_list = []):
        """
        Call training sequence if buffer not emtpy
        """
        if buffer_list==[]:
            return
        curr_state_, action, n_step_return = zip(*buffer_list)
        self.training( curr_state_, action, n_step_return)
    

    def update_experience_buffer(self, buffer_list):
        """
        selects samples for replay buffer
        """
        list_1 =[ [x,y,z] for x,y,z in sorted(buffer_list, key=lambda pair: pair[2],reverse=True) if z > self.buffer_threshold] 
       
        list_2 =[ [x,y,z] for x,y,z in buffer_list if z > self.buffer_threshold] 
        size_  = min(int(len(list_2)/2),int(self.replay_buffer_size/2))
        buffer_list = random.sample(list_2,k=size_) + list_1[:size_]
     
        return buffer_list

    def update_epsilon(self,t):
        """
        Epsilon decay function§e
        """
        self.epsilon = max(1.0 - 1.50*float(t)/float(self.play_epochs),0.0)*(self.epsilon_start-self.epsilon_end) + self.epsilon_end
        #self.epsilon = max(1.0 - np.exp(float(t)-float(self.play_epochs)),0.0)*(self.epsilon_start-self.epsilon_end) + self.epsilon_end

class FlatConvert(FlatLink):

    class flatcplxdef():
        def __init__(self,size_x,size_y,agents,add_track,id_):
            self.size_x = size_x
            self.size_y = size_y
            self.agents = agents
            self.add_track = add_track
            self.id_ = id_
    
    class flatsparsdef():
        def __init__(self,size_x,size_y,agents,cities,intersect,stations,id_):
            self.size_x = size_x
            self.size_y = size_y
            self.agents = agents
            self.cities = cities
            self.intersect = intersect
            self.stations = stations
            self.id_ = id_

        
    def __init__(self,size_x,size_y):
        super(FlatConvert, self).__init__(size_x,size_y)

    @staticmethod
    def data_producer_sparse(this,lock,num_agents,size_x,size_y,id_,cities,intersect, stations):
        """
        Play games in flatland using sparse_rail_generator and calculates n-step return
        """
        from observation import RawObservation
        from flatland.envs.rail_env import RailEnv
        from flatland.envs.rail_generators import random_rail_generator, complex_rail_generator ,sparse_rail_generator
        flatland = FlatConvert(this.size_x,this.size_y)
        flatland.gpu_ = 0 
        flatland.init_model()
        flatland.epsilon_start = this.epsilon_start
        flatland.epsilon_end = this.epsilon_end
        flatland.play_epochs = this.play_epochs

        node_dist = int(min(size_x/4,20))
        geny = sparse_rail_generator(num_cities=cities, num_intersections=intersect, num_trainstations=stations, min_node_dist=node_dist, node_radius=2,
                          num_neighb=3, enhance_intersection=False, seed= id_*this.number_of_levels)
        env = RailEnv(size_x,size_y,rail_generator=geny, number_of_agents=num_agents, obs_builder_object=RawObservation([flatland.size_x,flatland.size_y]))
        total_games  = float(this.play_game_number*this.number_of_levels)
        best_reward = -9999999999.9
        for j in range(this.play_epochs):
            flatland.update_epsilon(j)
            reward_mean= 0.0
            predicted_mean = 0.0
            steps_mean = np.zeros((num_agents))
            solved_mean = 0.0
            for k in range(this.play_game_number):
                with lock:
                    if this.weights is not None:
                        best_model_wts = copy.deepcopy(this.weights)
                        flatland.model.load_state_dict(best_model_wts)

                for i in range(this.number_of_levels):
                    predicted_, reward_ ,steps_, done_= flatland.collect_training_data(env,None,this.queue)
                    reward_mean+=reward_
                    predicted_mean+=predicted_
                    solved_mean += done_
                    steps_mean += np.floor(steps_)

                env.num_resets = 0

            reward_mean /= total_games
            predicted_mean /= total_games
            steps_mean /= total_games
            solved_mean /= total_games
     
            print("ID:",id_, ", play epoch: ",j,", buffer size: ", this.queue.qsize(),", reward (mean): ",reward_mean,
            ", solved (mean): ",solved_mean, ", steps (mean): ",steps_mean,", predicted (mean): ",predicted_mean)
            
            env.num_resets = 0
        
                

    @staticmethod
    def data_producer_cmplx(this,lock,num_agents,size_x,size_y,id_,nr_extra=5, min_dist=10):
        """
        Play games in flatland using complex_rail_generator and calculates n-step return
        """
        from observation import RawObservation
        from flatland.envs.rail_env import RailEnv
        from flatland.envs.rail_generators import random_rail_generator, complex_rail_generator
        flatland = FlatConvert(this.size_x,this.size_y)
        flatland.gpu_ = 0 
        flatland.init_model()
        flatland.epsilon_start = this.epsilon_start
        flatland.epsilon_end = this.epsilon_end
        flatland.play_epochs = this.play_epochs
        env = RailEnv(size_x,size_y,rail_generator=complex_rail_generator(num_agents,nr_extra=nr_extra, min_dist=min_dist,seed=id_*this.number_of_levels), \
        number_of_agents=num_agents,obs_builder_object=RawObservation([flatland.size_x,flatland.size_y]))
        total_games  = float(this.play_game_number*this.number_of_levels)
        best_reward = -9999999999.9
        for j in range(this.play_epochs):
            flatland.update_epsilon(j)
            reward_mean= 0.0
            predicted_mean = 0.0
            steps_mean = np.zeros((num_agents))
            solved_mean = 0.0
            for k in range(this.play_game_number):
                with lock:
                    if this.weights is not None:
                        best_model_wts = copy.deepcopy(this.weights)
                        flatland.model.load_state_dict(best_model_wts)

                for i in range(this.number_of_levels):
                    predicted_, reward_ ,steps_, done_= flatland.collect_training_data(env,None,this.queue)
                    reward_mean+=reward_
                    predicted_mean+=predicted_
                    solved_mean += done_
                    steps_mean += np.floor(steps_)

                env.num_resets = 0

            reward_mean /= total_games
            predicted_mean /= total_games
            steps_mean /= total_games
            solved_mean /= total_games
     
            print("ID:",id_, ", play epoch: ",j,", buffer size: ", this.queue.qsize(),", reward (mean): ",reward_mean,
            ", solved (mean): ",solved_mean, ", steps (mean): ",steps_mean,", predicted (mean): ",predicted_mean)
            
            env.num_resets = 0

    
            
    @staticmethod
    def data_consumer(this,lock):
        """
        Polls n-step returns from producer and trains model
        """
        import time
        flatland = FlatConvert(this.size_x,this.size_y)
        flatland.gpu_ = 0 
        flatland.init_model()
        buffer_list=[]
        epoch = 0
        while True:
 
            b_list = []

            while len(b_list) < flatland.buffer_size:
                b_list.append(this.queue.get())

            buffer_list =  buffer_list  + b_list
            flatland.trainer(buffer_list)
            buffer_list =  flatland.update_experience_buffer(buffer_list)
            with lock:
                if flatland.weights is not None:
                    this.weights = copy.deepcopy(flatland.weights)

            epoch+=1
            print("Trained epoch: ",epoch)
            
    
            
    def asynchronous_training(self,size_x,size_y,num_agents):
        """
        Run simulation with producer-consumer in training mode
        """
        self.queue = multiprocessing.Queue()
        buffer_list=[]
        best_reward = -999999999

        print("start training...")
     
        prod_1 = [
            self.flatsparsdef(20,20,5,5,2,5,1),
            self.flatsparsdef(40,40,10,8,15,12,2),
            self.flatsparsdef(30,30,8,10,20,18,3),
            self.flatsparsdef(30,30,8,16,20,12,4),
            self.flatsparsdef(40,40,20,10,20,25,5),
            self.flatsparsdef(30,30,10,16,15,10,6)]

        prod_2 = [
            self.flatcplxdef(20,20,5,2,7),
            self.flatcplxdef(30,30,5,2,8),
            self.flatcplxdef(30,30,8,4,9),
            self.flatcplxdef(80,80,1,20,10)]
       
        cons = [1]
        # Create a lock object to synchronize resource access
        lock = multiprocessing.Lock()
    
        producers = []
        consumers = []
    
        for n in prod_1:
            # Create our producer processes by passing the producer function and it's arguments
            p = multiprocessing.Process(target=FlatConvert.data_producer_sparse, args=(self, lock,n.agents,n.size_x,n.size_y,n.id_,n.cities,n.intersect,n.stations))
            p.daemon = True
            producers.append(p)
        for n in prod_2:
            # Create our producer processes by passing the producer function and it's arguments 
            p = multiprocessing.Process(target=FlatConvert.data_producer_cmplx, args=(self, lock,n.agents,n.size_x,n.size_y,n.id_,n.add_track,5))
            p.daemon = True
            producers.append(p)
        # Create consumer processes
        for i in cons:
            p = multiprocessing.Process(target=FlatConvert.data_consumer, args=(self, lock))
            
            # This is critical! The consumer function has an infinite loop
            # Which means it will never exit unless we set daemon to true
            p.daemon = True
            consumers.append(p)
    
        # Start the producers and consumer
        # The Python VM will launch new independent processes for each Process object
        for p in producers:
            p.start()
    
        for c in consumers:
            c.start()
    
        # Like threading, we have a join() method that synchronizes our program
        try:
            for p in producers:
                p.join()
        except KeyboardInterrupt:
            print('parent received ctrl-c')
            for p in producers:
                p.terminate()
                p.join()
        except:
            print('parent term')
            for p in producers:
                p.terminate()
                p.join()

        print('Parent process exiting...')

            
    def perform_training(self,env,env_renderer=None):
        """
        Run simulation in training mode
        """

        queue_ = queue.Queue()
        buffer_list=[]
        best_reward = -999999999
        best_solution = 0.0
        self.load_()
        #buffer_list=list(np.load('bufferdump.npy'))
        print("start training...")
        number_of_agents = len(env.agents)
        for j in range(self.play_epochs):
            self.update_epsilon(j)
            reward_mean= 0.0
            predicted_mean = 0.0
            steps_mean = np.zeros((number_of_agents))
            solved_mean = 0.0
            b_list=[]
            total_games  = float(self.play_game_number*self.number_of_levels)
            time_a = time.time()
            for k in range(self.play_game_number):
                for i in range(self.number_of_levels):
                    predicted_, reward_ ,steps_, done_= self.collect_training_data(env,env_renderer,queue_)
                    reward_mean+=reward_
                    predicted_mean+=predicted_
                    solved_mean += done_
                    steps_mean += np.floor(steps_)
                #list_ = list(self.queue.queue) 
                env.num_resets = 0
                #b_list = b_list + list_ #+ [ [x,y,z] for x,y,z,d in random.sample(list_,count_)]
            time_a = time.time()-time_a
            b_list = list(queue_.queue)
            reward_mean /= total_games
            predicted_mean /= total_games
            steps_mean /= total_games
            solved_mean /= total_games

            time_b = time.time()
            self.trainer(buffer_list + b_list)
            time_b = time.time()-time_b

            print("play epoch: ",j,", experience buffer size: ", len(buffer_list),", buffer size: ", len(b_list),", reward (mean): ",reward_mean,
            ", solved (mean): ",solved_mean, ", steps (mean): ",steps_mean,", predicted (mean): ",predicted_mean,"\n time used for training: ",
            time_b," time used for collect data: ",time_a)
           
            
           #allow warmup phase
            buffer_list =  self.update_experience_buffer(buffer_list + b_list)

            
            if reward_mean > best_reward:
                best_reward = reward_mean
                try:
                    torch.save(self.model, 'model_a3c.pt')
                    np.save('bufferdump', np.array(buffer_list))
                  
                except:
                    print("Model is not saved!!!")

            if solved_mean > best_solution:
                best_solution = solved_mean
                try:
                    torch.save(self.model, 'model_a3c_s.pt')
                  
                except:
                    print("Model solution is not saved!!!")
            queue_.queue.clear()
            
    def check_end_of_game(self,states_old, states_new, iter_):
        count = 0
        for key in states_old.keys():
            if np.array_equal(states_old[key][0],states_new[key][0]):
                count += 1

        if count == len(states_old) and iter_> 1:
            return True

        return False


    
    def shape_reward(self,states,reward):
        """
        Additional penalty for not moving (e.g when in deadlock), only applied on last time step!
        """

        
        states_=states[0].reshape((self.time_stack,int(states[0].shape[0]/self.time_stack),states[0].shape[1],states[0].shape[2]))
        #states_=states_[0]
        #states_ = states_.reshape((self.time_stack,int(states_.shape[1]/states_.shape[2]),states_.shape[2],states_.shape[2]))
        x = self.time_stack
        
        """
               {0: 'N',
                1: 'E',
                2: 'S',
                3: 'W'} direction_ = float(dir_ + 1)/5.0
        self.observation_space = np.stack(( smap,agent_positions_,agent_targets_,my_target_))
        ,[0.0,0.0,0.0,dir_,position[0],position[1]]
        return [self.observation_space, [vec_/dist_, dist_,dir_,position]]
        """
        n = float(1)/10.0
        e = float(2)/10.0
        s = float(3)/10.0
        w = float(4)/10.0
 
        my_pos_x = int(self.size_x/2)
        my_pos_y = int(self.size_y/2)
        my_dir = states[1][5] #states_[x-1,2,my_pos_x,my_pos_y]
        index_agent_target = 2
        if my_pos_x - 1 >=0:
            if states_[x-1,0,my_pos_x - 1,my_pos_y]>0.0 and states_[x-1,1,my_pos_x - 1,my_pos_y] == e \
            and my_dir!=e:
                reward += -2.0
                #print("blocking east")
            #if np.array_equal(states_[x-1,0,:,:],states_[x-2,0,:,:]) and states_[x-1,index_agent_target,my_pos_x-1,my_pos_y]>0.0:
            #    reward += -2.0

        if my_pos_x + 1 < self.size_x:
            if states_[x-1,0,my_pos_x + 1,my_pos_y]>0.0 and states_[x-1,1,my_pos_x + 1,my_pos_y] == w \
            and my_dir!=w:
                reward += -2.0
                #print("blocking west")  
            #if np.array_equal(states_[x-1,0,:,:],states_[x-2,0,:,:]) and states_[x-1,index_agent_target,my_pos_x+1,my_pos_y]>0.0:
            #    reward += -2.0   
        if my_pos_y - 1 >=0:
            if states_[x-1,0,my_pos_x,my_pos_y-1]>0.0 and states_[x-1,1,my_pos_x,my_pos_y-1] == s \
            and my_dir!=s:
                reward += -2.0
                #print("blocking south")
            #if np.array_equal(states_[x-1,0,:,:],states_[x-2,0,:,:]) and states_[x-1,index_agent_target,my_pos_x,my_pos_y-1]>0.0 :
            #    reward += -2.0   
        if my_pos_y + 1 < self.size_y:
            if states_[x-1,0,my_pos_x,my_pos_y+1]>0.0 and states_[x-1,1,my_pos_x,my_pos_y+1] == n \
            and my_dir!=n:
                reward += -2.0
                #print("blocking nord")
            #if np.array_equal(states_[x-1,0,:,:],states_[x-2,0,:,:]) and states_[x-1,index_agent_target,my_pos_x,my_pos_y+1]>0.0 :
            #    reward += -2.0   
        
        if states_[x-1,index_agent_target,my_pos_x,my_pos_y] >= 0.0: #new value and pos!
            reward += -5.0
            #print("target hijack")
        
        
        return reward

    def update_coordinates(self,state_,action_):
        return 0

    def make_virtual_step(self, state_dict):

        for agent_id, state in curr_state_dict.items():
                    policy, value = self.predict(state)
                    policy_dict[agent_id] = policy
                    value_dict[agent_id] = value
                    action_dict[agent_id], pre = self.get_action(policy)


    def make_state(self,state_):
        """
        Stack states 3 times for initial call (t1==t2==t3)
        """
        state = state_[0] #target does not move!
        state_tuple = np.stack(tuple([state]*self.time_stack))
        state_tuple = np.stack(state_tuple).reshape((self.time_stack*state.shape[0],state.shape[1],state.shape[2]))
        return [state_tuple,state_[1]]

    def update_state(self,states_a,state):
        """
        update state by removing oldest timestep and adding actual step
        """
        
        states_ = np.copy(states_a[0])
        states_=states_.reshape((self.time_stack,state[0].shape[0],state[0].shape[1],state[0].shape[2]))
      
        states_[:self.time_stack-1,:,:,:] =  states_[1:,:,:,:]
        states_[self.time_stack-1,:,:,:] = np.copy(state[0]).reshape((1,state[0].shape[0],state[0].shape[1],state[0].shape[2]))
        states_ = states_.reshape((self.time_stack*state[0].shape[0],state[0].shape[1],state[0].shape[2]))
      
        return [states_,np.copy(state[1])]


    def init_state_dict(self,state_dict_):
        for key in state_dict_.keys():
            state_dict_[key] = self.make_state(state_dict_[key])
        return state_dict_

    def update_state_dict(self,old_state_dict, new_state_dict):
        for key in old_state_dict.keys():
            new_state_dict[key] = self.update_state(old_state_dict[key],new_state_dict[key])
        return new_state_dict

    def play(self,env, filename , env_renderer, epochs_ = 10):
        """
        Run simulation on trained model
        """
        episode_reward = 0
        self.thres = 0.0
        self.load_(filename)
        self.set_gpu()
        self.model.eval()
        global_reward = 0
        for i in range(epochs_):
            if env_renderer is not None:   
                env_renderer.reset()
            policy_dict = {}
            value_dict = {}
            action_dict = {}
            #self.max_iter = 40
            iter_ = 0
            pre_ = 0
            act_ = 0
            done = False
            curr_state_dict = self.init_state_dict(env.reset())
            #curr_state_dict = env.reset()
            while not done and iter_< self.max_iter:
                iter_+=1
                for agent_id, state in curr_state_dict.items():
                    policy, value = self.predict(state)
                    policy_dict[agent_id] = policy
                    value_dict[agent_id] = value
                    action_dict[agent_id], pre = self.get_action(policy)
                    pre_ += pre
                    act_+=1 

                if env_renderer is not None:    
                    env_renderer.render_env(show=True,show_observations=False)
               
                next_state_dict, reward_dict, done_dict, _ = env.step(action_dict)
                
                next_state_dict = self.update_state_dict(curr_state_dict, next_state_dict)
                done = done_dict['__all__']
                
                curr_state_dict = next_state_dict

        return  float(pre_)/float(act_), global_reward, float(iter_)/float(self.max_iter)

    def step(self,obs_):
        """
        Perform single step, useful for submission
        """

        if self.state is None:
            self.state = self.init_state_dict(obs_)
        else:
            self.state = self.update_state_dict(self.state, obs_)

        value_dict = {}
        action_dict = {}
        for agent_id, state in state_dict.items():
            policy, value = self.predict(self.state)
            value_dict[agent_id] = value
            action_dict[agent_id], pre = self.get_action(policy)

        return action_dict

       

    def collect_training_data(self,env,env_renderer,queue):
        """
        Run single simualtion szenario for training
        """
        done = False
        curr_state_dict = self.init_state_dict(env.reset())
        #curr_state_dict = env.reset()
        episode_reward = 0
        memory={}
        policy_dict = {}
        for agent_id, state in curr_state_dict.items():
            memory[agent_id] = []
            policy_dict[agent_id] = np.zeros((self.model.num_actions_))
        self.model.eval()
        
        value_dict = {}
        action_dict = {}
        iter_counts = [0] * len(curr_state_dict)
        finished = [False] * len(curr_state_dict)
        iter_ = 0
        pre_ = 0
        act_ = 0
        global_reward = 0.0
        if env_renderer is not None:
            env_renderer.reset()
        while not done and iter_< self.max_iter:
            iter_+=1
            for agent_id, state in curr_state_dict.items():
                if not finished[agent_id]:
                    policy, value = self.predict(state)
                    policy_dict[agent_id] += policy
                    value_dict[agent_id] = value
                    action_dict[agent_id], pre = self.get_action(policy)
                    pre_ += pre
                    act_+=1 
                
            if env_renderer is not None:    
                env_renderer.render_env(show=True,show_observations=False)

            next_state_dict, reward_dict, done_dict, _ = env.step(action_dict)
            next_state_dict = self.update_state_dict(curr_state_dict, next_state_dict)

            #reward_dict = self.modify_reward(reward_dict,done_dict,next_state_dict)

            done = done_dict['__all__']


                
            for agent_id, state in curr_state_dict.items():
                
                if not finished[agent_id]:
                    global_reward += reward_dict[agent_id]
                    reward_dict[agent_id] = self.shape_reward(next_state_dict[agent_id], reward_dict[agent_id])
                    #curr_state, action, reward, next_state, done
                    memory[agent_id].append(tuple([curr_state_dict[agent_id], 
                    action_dict[agent_id], reward_dict[agent_id], next_state_dict[agent_id], done_dict[agent_id],value_dict[agent_id]]))
                    """
                    done_ = done_dict[agent_id]
                    while ((len(memory[agent_id]) >= self.n_step_return_) or (done_ and not memory[agent_id] == [])):
                        curr_state, action, n_step_return, memory[agent_id], _ = self.accumulate(memory[agent_id],value_dict[agent_id], done)
                        queue.put([curr_state, action, n_step_return])
                    """
                    
                    iter_counts[agent_id] += 1
                    
                    if done_dict[agent_id] is True:
                       
                        finished[agent_id] = True
            
      
            curr_state_dict = next_state_dict
        
        for agent_id, state in curr_state_dict.items():
            
            done_ = done_dict[agent_id]
            while ((len(memory[agent_id]) >= self.n_step_return_) or (done_ and not memory[agent_id] == [])):
                    curr_state, action, n_step_return, memory[agent_id], _ = self.accumulate(memory[agent_id],0, done)
                    queue.put([curr_state, action, n_step_return])
        
        
        return  float(pre_)/float(act_), global_reward, np.array(iter_counts).astype(np.float)/float(self.max_iter), 1 if done else 0
