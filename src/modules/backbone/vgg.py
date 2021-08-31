import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .src.modules.Module import Module
VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGGNet(Module):

    def __init__( self, input_layer, input_shape, include_head=None ,vggtype="VGG16"):
        r"""
        input_layer -> input channels -> how many channels the input is
        input_shape -> input image of shape (H,W)
        include_head -> if None then just use the backbone if yes use the the head also for application classification
        """
        super().__init__()
        self.in_channels = input_layer
        self.conv_layers = self.create_conv_layers(VGG_types[vggtype])
        self.tfilter = 0
        self.include_head = include_head
        if include_head is not None:
            arsi = VGG_types[vggtype]
            padding_list = [1 if d != "M" else 0 for d in arsi]
            kernel_list = [3 if d != "M" else 2 for d in arsi]
            stride_list = [1 if d != "M" else 2 for d in arsi]
            self.output_size = input_shape
            for  i in range(len(arsi)):
                self.output_size[0] = int((math.floor((self.output_size[0] + 2*padding_list[i] - kernel_list[i])/stride_list[i] +1)))
                self.output_size[1] = int((math.floor((self.output_size[1] + 2*padding_list[i] - kernel_list[i])/stride_list[i] +1)))


            self.fcs = nn.Sequential(
                
                nn.Linear(self.output_size[0]*self.output_size[1]*512 , 4096),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(4096, include_head),
            )

    def forward(self, x):
        x = self.conv_layers(x)
   
        if self.include_head is not None:
            #x = torch.flatten(x)
            x = x.reshape(x.shape[0], -1)
            x = self.fcs(x)

        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3),
                        stride=(1),
                        padding=(1),
                    ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(),
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2), stride=(2))]

        return nn.Sequential(*layers)

def VGG11(input_layer,input_shape, include_head=None):
    return VGGNet(input_layer,input_shape,include_head,"VGG11")

def VGG13(input_layer,input_shape, include_head=None):
    return VGGNet(input_layer,input_shape,include_head,"VGG13")

def VGG16(input_layer,input_shape, include_head=None):
    return VGGNet(input_layer,input_shape,include_head,"VGG16")

def VGG19(input_layer,input_shape, include_head=None):
    return VGGNet(input_layer,input_shape,include_head,"VGG19")

