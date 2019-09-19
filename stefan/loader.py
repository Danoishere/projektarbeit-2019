"""
Author: S.Huschauer
Date: 31.08.2019
Email: huschste@students.zhaw.ch
Associated with ZHAW datalab
This code is not exhaustive!
"""

import torch
import torch.nn as nn
from torchvision import utils 
import numpy as np

class FlatData(torch.utils.data.dataset.Dataset):
    """  
    Prepare data to read batch-wise, create dataset

    """
    def __init__(self,states,targets,rewards,num_action,device,dtype):
        self.states_ = [torch.from_numpy(x[0].astype(np.float))  for x in states]
        self.states_2_ = [torch.from_numpy(x[1].astype(np.float))  for x in states]
        self.targets_ = [torch.from_numpy(np.eye(num_action)[x])  for x in targets]
        self.rewards_ = [torch.from_numpy(np.array([x]).reshape(-1)) for x in rewards]
       

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        
        return self.states_[index], self.states_2_[index], self.targets_[index], self.rewards_[index]

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.states_)

