"""
reference "https://github.com/facebookresearch/detr"
"""


import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from .detr.datasets import coco
from .detr.util import misc as utils
from .detr.datasets import build_dataset, get_coco_api_from_dataset
from .detr.engine import evaluate, train_one_epoch
from .detr.models import build_model

class DETR(torch.nn.Module):
    def __init__(self, 
                type:str,
                num_classes:int=91, 
                enc_layers:int=6, 
                dec_layers:int=6, 
                dim_feedforward:int=2048, 
                hidden_dim:int=256, 
                dropout:float=0.1, 
                nheads:int=8,
                position_embedding:str='sine', 
                num_queries:int=100, 
                pre_norm:bool=False,
                seed:int=42,
                aux_loss:bool=True,
                masks:bool=False,
                panoptic:bool=False,
                pretrained:str=None,
                bbox_loss_coef:float=5,
                giou_loss_coef:float=2,
                dice_loss_coef:float=1,
                mask_loss_coef:float=1,
                eos_coef:float=0.1,
                lr_backbone:float=1e-5,
                frozen_weights:str=None,
                set_cost_class:float=1,
                set_cost_bbox:float=5,
                set_cost_giou:float=2):
        ##Parameters Validation##
        if type not in ['detr-r50','detr-dc5-r50','detr-r101','detr-dc5-r101']:
            raise ValueError("type only support on 'detr-r50','detr-dc5-r50','detr-r101','detr-dc5-r101'")
        
        if type == 'detr-r50':
            backbone = 'resnet50'
            dilation = False
        elif type == 'detr-dc5-r50':
            backbone = 'resnet50'
            dilation = True
        elif type == 'detr-r101':
            backbone = 'resnet101'
            dilation = False
        else:
            backbone = 'resnet101'
            dilation = True

        if position_embedding not in ['sine', 'learned']:
            raise ValueError("position embedding only support on 'sine' and 'learned'")

        if frozen_weights is not None:
            assert masks, "Frozen training is meant for segmentation only"
        #####################################
        self.type = type
        self.pretrained = pretrained
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.num_classes = num_classes
        self.seed = seed + utils.get_rank()
        self.frozen_weights = frozen_weights
        self.lr_backbone = lr_backbone
        self.masks = masks
        self.panoptic = panoptic
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model, self.criterion, self.postprocessors = build_model(backbone, dilation, enc_layers, dec_layers,dim_feedforward, nheads, dropout,pre_norm,hidden_dim,position_embedding,num_classes, device, num_queries, aux_loss, masks, panoptic, frozen_weights, bbox_loss_coef, giou_loss_coef, dice_loss_coef, mask_loss_coef, eos_coef, lr_backbone, set_cost_class, set_cost_bbox, set_cost_giou) #need to fix the build model function
        self.model.to(device)

        self.n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def forward(self, x):
        return self.model(x)
    
    def fit(
        self,
        train_path:str,
        val_path:str,
        lr:float=1e-4,
        batch_size:int=2,
        num_workers:int=2,
        weight_decay:float=1e-4,
        start_epoch:int=1,
        epochs:int=300,
        lr_drop:int=200,
        clip_max_norm:float=0.1,
        output_dir:str=None
    ):
        param_dicts = [
        {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": self.lr_backbone,
        },
        ]#perlu dibuatin skema multi GPU
        optimizer = torch.optim.AdamW(param_dicts, lr=lr,
                                  weight_decay=weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_drop)

        dataset_train = build_dataset(image_set = 'train', path=train_path, masks=self.masks, panoptic=self.panoptic) #perlu dibuat skema untuk build datasets
        dataset_val = build_dataset(image_set = 'val',path=val_path, masks=self.masks, panoptic=self.panoptic)

        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=num_workers)
        data_loader_val = DataLoader(dataset_val, batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=num_workers)
        
        if self.panoptic:
            # We also evaluate AP during panoptic training, on original coco DS
            coco_val = coco.build(image_set = 'val',path=val_path, masks=self.masks)
            base_ds = get_coco_api_from_dataset(coco_val)
        else:
            base_ds = get_coco_api_from_dataset(dataset_val)
        
        if self.frozen_weights is not None:
            checkpoint = torch.load(self.frozen_weights, map_location='cpu')
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
                train_stats = train_one_epoch(
                    self.model, self.criterion, data_loader_train, optimizer, self.device, epoch,
                    clip_max_norm)
                lr_scheduler.step()
                if output_dir is not None:
                    checkpoint_paths = [output_dir/ 'checkpoint.pth']
                    if (epoch + 1) % lr_drop == 0 or (epoch + 1) % 100 == 0:
                        checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': self.model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                        }, checkpoint_path)
                test_stats, coco_evaluator = evaluate(
                    self.model, self.criterion, self.postprocessors, data_loader_val, base_ds, self.device, output_dir #need to fix later
                )
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': self.n_parameters}
                
                if output_dir is not None and utils.is_main_process():
                    with (output_dir / "log.txt").open("a") as f:
                        f.write(json.dumps(log_stats) + "\n")
            
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))
        
    def eval(
        self,
        test_path:str,
        batch_size:int=2,
        num_workers:int=2,
        output_dir:str=None
    ):
        dataset_val = build_dataset(image_set = 'val',path=test_path, masks=self.masks, panoptic=self.panoptic)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        data_loader_val = DataLoader(dataset_val, batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=num_workers)
        
        if self.panoptic:
            # We also evaluate AP during panoptic training, on original coco DS
            coco_val = coco.build(image_set = 'val',path=test_path, masks=self.masks)
            base_ds = get_coco_api_from_dataset(coco_val)
        else:
            base_ds = get_coco_api_from_dataset(dataset_val)

        test_stats, coco_evaluator = evaluate(self.model, self.criterion, self.postprocessors,
                                              data_loader_val, base_ds, self.device, output_dir)
        if output_dir is not None:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")