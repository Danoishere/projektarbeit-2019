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
class FlatNet(nn.Module):
    """
    Definition of a3c model, forward pass and loss.
    """
    def __init__(self,input_layer_size = 16,init_weights=True):
        super(FlatNet, self).__init__()
        self.observations_= 6406
        self.num_actions_= 5
        self.dense_layer_size = 256
        self.hidden_layer_size = 512
        self.lstm_layer_size  = 1
        self.avgpool2 = nn.AdaptiveAvgPool2d((5, 5))
        
        self.feat_1 = nn.Sequential(
            nn.Conv2d(input_layer_size, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
         
        )
        self.feat_1b = nn.Sequential(
            nn.Conv2d(input_layer_size, 32, kernel_size=1, padding=0),
           
        )
        self.feat_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
         
        )
        self.feat_2b = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, padding=0),
          
        )
        self.classifier_ = nn.Sequential( 
            nn.Linear(self.observations_,  self.hidden_layer_size),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(self.hidden_layer_size,  self.dense_layer_size),
            nn.ReLU(True),
            nn.Dropout(),
        )
    
        self.lstm_ = nn.LSTM(self.dense_layer_size, self.dense_layer_size, self.lstm_layer_size) 
        self.policy_ = nn.Sequential(
            nn.Linear(self.dense_layer_size, self.num_actions_),
            nn.Softmax(dim=1),
        )
        self.value_ = nn.Sequential(
            nn.Linear(self.dense_layer_size,  self.dense_layer_size),
            nn.ReLU(True),
            nn.Linear(self.dense_layer_size,1)
            )


        if init_weights==True:
            self._initialize_weights(self.classifier_)

            self._initialize_weights(self.policy_)
            self._initialize_weights(self.value_)
            self._initialize_weights(self.feat_1)
            self._initialize_weights(self.feat_1b)
            self._initialize_weights(self.feat_2)
            self._initialize_weights(self.feat_2b)

    
    def features_1(self,x):
        x1 = self.feat_1(x)
        x1b = self.feat_1b(x)
        return torch.cat([x1,x1b],1)

    def features_2(self,x):
        x2 = self.feat_2(x)
        x2b = self.feat_2b(x)
        x = torch.cat([x2,x2b],1)
        return x 
 
    def features(self,x):
        x = F.relu(self.features_1(x),True)
        x = F.max_pool2d(x,kernel_size=2, stride=2)
        #x = F.dropout2d(x)
        x = F.relu(self.features_2(x),True)
        x = F.max_pool2d(x,kernel_size=2, stride=2)
        #x = F.dropout2d(x)
        x = self.avgpool2(x)
        return x

    def forward(self, x, x2):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x2 = x2.view(x2.size(0),-1)
        x = torch.cat([x,x2],1)
        x = self.classifier_(x)
        x, _ = self.lstm_(x.view(len(x), 1, -1))
        x = x.view(len(x), -1)
        p = self.policy_(x)
        v = self.value_(x)
        return p,v

    def _initialize_weights(self,modules_):
        for i,m in enumerate(modules_):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.1, 0.3)
                nn.init.constant_(m.bias, 0)
    