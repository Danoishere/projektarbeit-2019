"""
Author: S.Huschauer
Date: 31.08.2019
Email: huschste@students.zhaw.ch
Associated with ZHAW datalab
This code is not exhaustive and "is as it is"!
"""
import torch.nn as nn
import torch

import torch.nn.functional as F

class FlatLoss(nn.Module):
    def __init__(self):
        #super(FlatNet, self).__init__()
        self.loss_value_ = 0.5
        self.loss_entropy_ = 0.01
        self.loss_obs = 0.01
        self.epsilon_ = 1.e-12
    
    def loss(self,out_policy, y_policy, out_values, y_values, obs_2):
        y_policy = y_policy.requires_grad_(requires_grad=True)
        y_values = y_values.requires_grad_(requires_grad=True)
        #dist_to_target =  obs_2[:,2].requires_grad_(requires_grad=True)
        log_prob = torch.log(torch.sum(out_policy*y_policy, 1)+self.epsilon_); 
        advantage = (y_values - out_values).flatten()
        policy_loss = -log_prob * advantage.detach()
        value_loss = self.loss_value_ * advantage.pow(2)
        entropy = self.loss_entropy_ * torch.sum(y_policy * torch.log(y_policy+self.epsilon_), 1)
        loss = torch.mean(policy_loss + value_loss + entropy )#+ dist_to_target*self.loss_obs)
        return  loss