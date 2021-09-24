import torch.nn as nn
import torch.nn.functional as F

from src.modules.module import Module
from src.modules.sequential import Sequential

#reference https://github.com/developer0hye/PyTorch-Darknet53/blob/master/model.py
#reference https://www.researchgate.net/figure/Structure-of-the-Darknet53-convolutional-network_fig4_338121987 see the arcitecture in there

def conv_batch(input_layer, output_layer, kernel_size=3, padding=1, stride=1):
    r"""
    a single Convolution Batch on Darknet53 that contain Conv2d, BatchNorm2d, LeakyReLU

    input:
    input_layer: int, how many layer the input is
    output_layer: int, how many layer the output is gonna be
    kernel_size:int, the kernel_size of single convolutions
    padding:int, padding the input with certain values
    stride: int, how many stride that gonna use on single convolution

    output:
    A Sequence model of Conv2d, BatchNorm2d, and LeakyRelu
    """
    return Sequential([
        nn.Conv2d(input_layer,output_layer, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(output_layer),
        nn.LeakyReLU()
    ])

class DarkResidualBlock(Module):
    r"""
    a single Residual Block
    input:
    input_layer: int, how many layer the input is

    output:
    a Residual block you can see the image on link above
    """
    def __init__(self, input_layer):
        super().__init__()

        reduced_channels = int(input_layer/2)

        self.conv1 = conv_batch(input_layer, reduced_channels, kernel_size=1, padding=0)
        self.conv2 = conv_batch(reduced_channels, input_layer)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out += residual

        return out

class Darknet53(Module):
    r"""
    a Darknet 53 architecture
    input:
    input_layer: int, how many layer the input is
    input_shape: list, H,W format for the input
    include_head: int/None, if None then the model is just used for backbone only, but if not None the model can used for classification

    output:
    Darknet53 model that can use wether just for backbone or for classifications
    """
    def __init__(self, input_layer, input_shape, include_head=None):
        super().__init__()

        self.include_head = include_head

        self.conv1 = conv_batch(input_layer,32)
        self.conv2 = conv_batch(32, 64, stride=2)

        self.residual_block1 = self.make_layer(DarkResidualBlock, input_layer=64, num_blocks=1)
        self.conv3 = conv_batch(64,128, stride=2)

        self.residual_block2 = self.make_layer(DarkResidualBlock, input_layer=128, num_blocks=2)
        self.conv4 = conv_batch(128,256, stride=2)

        self.residual_block3 = self.make_layer(DarkResidualBlock, input_layer=256, num_blocks=8)
        self.conv5 = conv_batch(256,512, stride=2)

        self.residual_block4 = self.make_layer(DarkResidualBlock, input_layer=512, num_blocks=8)
        self.conv6 = conv_batch(512,1024, stride=2)

        self.residual_block5 = self.make_layer(DarkResidualBlock, input_layer=1024, num_blocks=4)

        if self.include_head is not None:
            self.global_average_pool = nn.AdaptiveMaxPool2d((1,1))
            self.fc = nn.Linear(1024, self.include_head)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.residual_block1(x)
        x = self.conv3(x)

        x = self.residual_block2(x)
        x = self.conv4(x)

        x = self.residual_block3(x)
        x = self.conv5(x)

        x = self.residual_block4(x)
        x = self.conv6(x)

        x = self.residual_block5(x)

        if self.include_head is not None:
            x = self.global_average_pool(x)
            x = x.view(-1, 1024)
            x = self.fc(x)
        
        return x

    def make_layer(self, block, input_layer, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(block(input_layer))
        return Sequential(layers)