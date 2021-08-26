import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.Module import Module

class AlexNet(Module):
    def __init__(self, input_layer):
        super().__init__()
        self.input_layer = input_layer
        self.conv1 = nn.Conv2d(input_layer,96, (11,11), stride=4)
        self.pool = nn.MaxPool2d((3,3), stride=2)
        self.conv2 = nn.Conv2d(96, 256, (5,5), padding=2)
        self.conv3 = nn.Conv2d(256, 384, (3,3), padding=1)
        self.conv4 = nn.Conv2d(384, 384, (3,3), padding=1)
        self.conv5 = nn.Conv2d(384, 256, (3,3), padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x = self.pool(x)

        return x
