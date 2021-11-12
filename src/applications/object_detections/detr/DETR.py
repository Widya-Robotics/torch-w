"""
reference "https://github.com/facebookresearch/detr"
"""


import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from .util import misc as utils
from .datasets import build_dataset, get_coco_api_from_dataset
from .engine import evaluate, train_one_epoch
from .models import build_model

class DETR(torch.nn.Module):
    def __init__(self, 
                type:str,
                num_classes:int=91, 
                enc_layers:int=6, 
                dev_layers:int=6, 
                dim_feedforward:int=2048, 
                hidden_dim:int=256, 
                dropout:float=0.1, 
                nheads:int=8, 
                num_queries:int=100, 
                pre_norm:bool=False,
                seed:int=42,
                pretrained:str=None):

        if type not in ['detr-r50','detr-dc5-r50','detr-r101','detr-dc5-r101']:
            raise ValueError("type only support on 'detr-r50','detr-dc5-r50','detr-r101','detr-dc5-r101'")
        self.type = type
        self.pretrained = pretrained
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        self.seed = seed + utils.get_rank()
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model, criterion, postprocessors = build_model() #need to fix the build model function
        self.model.to(device)
    
    def forward(self, x):
        return self.model(x)
    
    def fit(
        self,
        train_path:str,
        val_path:str,
        lr:float=1e-4,
        lr_backbone:float=1e-5,
        batch_size:int=2,
        num_workers:int=2,
        weight_decay:float=1e-4,
        start_epoch:int=1,
        epochs:int=300,
        lr_drop:int=200,
        clip_max_norm:float=0.1,
        frozen_weights:str=None,
    ):
        param_dicts = [
        {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": lr_backbone,
        },
        ]#perlu dibuatin skema multi GPU
        optimizer = torch.optim.AdamW(param_dicts, lr=lr,
                                  weight_decay=weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_drop)

        dataset_train = build_dataset() #perlu dibuat skema untuk build datasets
        dataset_val = build_dataset()

        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=num_workers)
        data_loader_val = DataLoader(dataset_val, batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=num_workers)
        
        if frozen_weights is not None:
            checkpoint = torch.load(frozen_weights, map_location='cpu')
            self.detr.load_state_dict(checkpoint['model'])
        
        if self.pretrained is not None:
            if self.pretrained.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    self.pretrained, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(self.pretrained, map_location='cpu')
            self.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                start_epoch = checkpoint['epoch'] + 1
            
            print("===========!!!Start Training!!!===========")
            start_time = time.time()
            for epoch in range(start_epoch, epochs):
                pass