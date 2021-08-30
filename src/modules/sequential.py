import torch
import torch.nn as nn
from src.modules.module import Module
from collections import OrderedDict

class Sequential(Module):
    r"""
    Sequential module for stacking the Neural Network architecture
    inherit the class Module
    input:
    models: type list or OrderedDict (list of models e.g: list -> [nn.Conv2d(), nn.MaxPool2d(), nn.Conv2d],
                                                          OrderedDict -> OrderedDict([
                                                                        ('conv1', nn.Conv2d(1,20,5)),
                                                                        ('relu1', nn.ReLU()),
                                                                        ('conv2', nn.Conv2d(20,64,5)),
                                                                        ('relu2', nn.ReLU())
                                                                        ]))
    """
    def __init__(self, models):
        super().__init__()
        if isinstance(models, list):
            self.models = nn.Sequential(*models)
        elif isintance(models, OrderedDict):
            self.models = nn.Sequential(models)
        else:
            raise ValueError('only accept list or OrderedDict type of data')
    
    def forward(self, x):
        r"""
        forward the input to the models
        """
        x = self.models(x)

        return x

    def __len__(self):
        r"""
        get the len of the models
        """
        return len(self.models)

    def __getitem__(self, idx):
        r"""
        get the item of the modules
        """
        return self.models[idx]

    def __iter__(self):
        return iter(self.models)