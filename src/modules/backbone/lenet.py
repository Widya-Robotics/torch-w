import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.Module import Module
import math


class LeNet(Module):
    r"""
    this backbone is implementation of first Convolution Neural Network by Yan Le Cun 
    """
    def __init__(self, input_layer, input_shape, include_head=None):
        r"""
        input_layer -> input channels -> how many channels the input is
        input_shape -> input image of shape (H,W)
        include_head -> if None then just use the backbone if yes use the the head also for application classification
        """
        super().__init__()
        self.input_layer = input_layer
        
        self.conv1 = nn.Conv2d(input_layer, 6, (5,5), 1, 2)
        self.avg_pool = nn.AvgPool2d((2,2), 2)
        self.conv2 = nn.Conv2d(6, 16, (5,5), 1)

        self.include_head = include_head

        if include_head is not None:
            padding_list = [2,0,0,0]
            kernel_list = [5,2,5,2]
            stride_list = [1,2,1,2]
            self.output_size = input_shape
            for  i in range(4):
                self.output_size[0] = int((math.floor((self.output_size[0] + 2*padding_list[i] - kernel_list[i])/stride_list[i] +1)))
                self.output_size[1] = int((math.floor((self.output_size[1] + 2*padding_list[i] - kernel_list[i])/stride_list[i] +1)))
            self.fc1 = nn.Linear(self.output_size[0]*self.output_size[1]*16, 120)
            self.fc2 = nn.Linear(120,84)
            self.fc3 = nn.Linear(84,include_head)
    
    def forward(self, x):
        x = F.sigmoid(self.conv1(x))
        x = self.avg_pool(x)

        x = F.sigmoid(self.conv2(x))
        x = self.avg_pool(x)
        # print(x.shape)

        if self.include_head is not None:
            x = torch.flatten(x)
            # print(x.shape)

            x = F.sigmoid(self.fc1(x))
            x = F.sigmoid(self.fc2(x))
            x = F.sigmoid(self.fc3(x))

        return x