from pyexpat import features
from re import X
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn


class LevelAttention(nn.Module):
    def __init__(self, channel):
        super(LevelAttention, self).__init__()

        self.AttnConv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, 1, kernel_size=1),
            nn.ReLU(),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.AttnConv(x)
        return x * x1



class SpatialAttention(nn.Module): 
    def __init__(self, kernel_size=7): 
        super(SpatialAttention, self).__init__() 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7' 
        padding = 3 if kernel_size == 7 else 1 
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False) 
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x): 
        avg_out = torch.mean(x, dim=1, keepdim=True) 
        max_out, _ = torch.max(x, dim=1, keepdim=True) # _索引,维度不变 
        x1 = torch.cat([avg_out, max_out], dim=1) 
        x1 = self.conv1(x1) 
        x1 = self.sigmoid(x1)
        return x * x1



class SELayer(nn.Module):
    def __init__(self,channel,reduction=16):
        super(SELayer,self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel,channel//16,bias=False),
            nn.ReLU(),
            nn.Linear(channel//16,channel,bias=False),
            nn.Sigmoid()
        )
        
    def forward(self,x):
    	#1,16,64,64
        b,c,_,_ = x.size()
        y = self.avg_pool(x).view(b,c)
        #1,16
        #print(y.size())
        y = self.fc(y).view(b,c,1,1)
        #1,16,1,1
        #print(y.size())
        #print(y.expand_as(x))
        #y.expand_as(x) 把y变成和x一样的形状
        return x * y.expand_as(x)



class LSC(nn.Module):
    def __init__(self, channel):
        super(LSC, self).__init__()

        self.la = LevelAttention(channel)
        self.sa = SpatialAttention(kernel_size=7)
        self.ca = SELayer(channel)
    
    def forward(self, x):

        LA = self.la(x)
        SA = self.sa(x)
        CA = self.ca(x)

        out = LA + SA + CA + x
        return out


