import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.modules.Module import Module

class AlexNet(Module):
    r"""
    AlexNet is the name of a convolutional neural network (CNN) architecture, designed by Alex Krizhevsky in collaboration 
    with Ilya Sutskever and Geoffrey Hinton, who was Krizhevsky's Ph.D. advisor

    AlexNet is build based on ImageNet classification challange
    """
    def __init__(self, input_layer, input_shape, include_head=None):
        super().__init__()
        self.input_layer = input_layer
        self.conv1 = nn.Conv2d(input_layer,96, (11,11), stride=4)
        self.pool = nn.MaxPool2d((3,3), stride=2)
        self.conv2 = nn.Conv2d(96, 256, (5,5), padding=2)
        self.conv3 = nn.Conv2d(256, 384, (3,3), padding=1)
        self.conv4 = nn.Conv2d(384, 384, (3,3), padding=1)
        self.conv5 = nn.Conv2d(384, 256, (3,3), padding=1)

        self.include_head = include_head
        
        if include_head is not None:
            padding_list = [0,0,2,0,1,1,1,0]
            kernel_list = [11,3,5,3,3,3,3,3]
            stride_list = [4,2,1,2,1,1,1,2]
            self.output_size = [0,0]
            for  i in range(8):
                self.output_size[0] = int((math.floor((input_shape[0] + 2*padding_list[i] - kernel_list[i])/stride_list[i] +1)))
                self.output_size[1] = int((math.floor((input_shape[1] + 2*padding_list[i] - kernel_list[i])/stride_list[i] +1)))
            self.fc1 = nn.Linear(self.output_size[0]*self.output_size[1]*256, 4096)
            self.dropout1 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(4096,4096)
            self.dropout2 = nn.Dropout(0.5)
            self.fc3 = nn.Linear(4096,include_head)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x = self.pool(x)

        if self.include_head is not None:
            x.view(-1, 256*self.output_size[0]*self.output_size[1])

            x = F.relu(self.fc1(x))
            x = self.dropout1(x)

            x = F.relu(self.fc2(x))
            x = self.dropout2(x)

            x = F.relu(self.fc3(x))

        return x
