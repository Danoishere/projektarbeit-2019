import numpy as np
"""
Author: S.Huschauer
Date: 31.08.2019
Email: huschste@students.zhaw.ch
Associated with ZHAW datalab
This code is not exhaustive and "is as it is"!
"""

class fakeEnv():
    """
    Dummy environment for mock test of a3c.
    """
    def __init__(self):
        self.grid_size = tuple([20,10])
        self.num_agents = 2
        self.done = False
        self.iterate = 0
        self.grid = np.zeros(self.grid_size)

    def reset(self):
        self.done = False
        self.iterate = 0
        grid_ = self.grid
        return grid_

    def step(self, action):
        grid_ = self.grid
        fake_rail =  np.random.rand(self.grid_size[0],self.grid_size[1])
        grid_[fake_rail<0.25]=0.25
        grid_[fake_rail>0.45]=0.5
        reward = np.array(np.mean(fake_rail)/self.grid_size[0]/self.grid_size[1])
        self.iterate+=1
        return grid_,reward,self.iterate > 100, None

