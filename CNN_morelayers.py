# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# More hidden layers input size (3,128,128)
# (3,64,64) remove last maxPool
#(3,32,32) remove first maxpool

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class NeuronalNetwork_deeper(nn.Module):

    def __init__(self):
        super(NeuronalNetwork_deeper, self).__init__()

        self.conv1 = nn.Conv2d(3,  32, 5 , padding=2)   #using stride=2 makes model worse #input dim, output dim, conv dim (5x5) , padding to reserve img size
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5,  padding= 2)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)  #using 2 times
        self.conv5_bn = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)  #using 2 times
        self.conv6_bn = nn.BatchNorm2d(256)
        self.conv7 = self.conv6
        self.conv7_bn = self.conv6_bn
        self.conv8 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv8_bn = nn.BatchNorm2d(512)


        self.fc1 = nn.Linear(512*1*1 , 120)  
        self.fc2 = nn.Linear(120, 12)  
        #self.out_layer = nn.Linear(120, 1)  #when use BCE
        self.out_layer = nn.Linear(12, 2)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))  #added batch norm, increaes total size rougthly +10 mb although decrease parameter 
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv6_bn(self.conv6(x)))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv7_bn(self.conv7(x)))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv8_bn(self.conv8(x)))
        #x = F.max_pool2d(x, (2, 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out_layer(x)
        x = self.softmax(x)
        return x
# -




