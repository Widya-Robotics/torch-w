import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.Module import Module


class LeNet(Module):
    r"""
    this backbone is implementation of first Convolution Neural Network by Yan Le Cun 
    """
    def __init__(self, input_layer):
        super().__init__()
        self.input_layer = input_layer
        
        self.conv1 = nn.Conv2d(input_layer, 6, (5,5), 1)
        self.avg_pool = nn.AvgPool2d((2,2), 2)
        self.conv2 = nn.Conv2d(6, 16, (5,5), 1)
        self.conv3 = nn.Conv2d(16,120, (5,5),1 )
    
    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.avg_pool(x))

        x = F.tanh(self.conv2(x))
        x = F.tanh(self.avg_pool(x))

        x = F.tanh(self.conv3(x))

        return x