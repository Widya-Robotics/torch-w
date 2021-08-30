import torch
from torch import nn
from src.modules.module import Module
from src.modules.sequential import Sequential

class MLP(Module):
    r"""
    Module MLP -> (Multi Layer Perceptron) module
    input: features -> list of features that gonna be created on the model.
    eg: 
    model = MLP([8,32,1024], input_shape=[10])
    output:
    model([
        nn.Linear(input_shape[-1], 8), nn.Linear(8,32), nn.Linear(32,1024)
    ])

    input: input_flatten -> if True then the input of the model gonna be flatten into 2d tensor (batch, feature) if false it's not gonna be flatten
    eg:
    model = MLP([8,32,1024], input_flatten=True)
    input_shape = (1,32,24,24) -> consider it's image after some convolution
    after_flatten = (1, 32*24*24) -> this input that gonna forward to the model
    """
    def __init__(self, features, input_shape, input_flatten=False):
        super().__init__()
        
        if not isinstance(features, list):
            raise ValueError('features only accept list type of data')
        if not (isinstance(input_shape, list) or isinstance(input_shape, tuple)):
            raise ValueError('input_shape only support list and tuple type of data')

        self.input_shape = input_shape
        self.input_flatten = input_flatten
        self.models = []

        if self.input_flatten:
            mul = 1
            for shp in input_shape:
                mul = mul*shp
            self.models.append(nn.Linear(mul, features[0]))
        
        else:
            self.models.append(nn.Linear(input_shape[-1], features[0]))

        for idx in range(len(features)-1):
            self.models.append(nn.Linear(features[idx], features[idx+1]))

        self.models = Sequential(self.models)
    
    def forward(self, x):
        if self.input_flatten:
            x = torch.flatten(x)

        x = self.models(x)

        return x