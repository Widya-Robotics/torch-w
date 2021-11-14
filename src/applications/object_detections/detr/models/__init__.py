# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build


def build_model(backbone, dilation,enc_layers, dec_layers,dim_feedforward, nheads, dropout,pre_norm,hidden_dim,position_embedding,num_classes, device, num_queries, aux_loss, masks, panoptic, frozen_weights, bbox_loss_coef, giou_loss_coef, dice_loss_coef, mask_loss_coef, eos_coef, lr_backbone, set_cost_class, set_cost_bbox, set_cost_giou):
    return build(backbone, dilation,enc_layers, dec_layers,dim_feedforward, nheads, dropout,pre_norm,hidden_dim,position_embedding,num_classes, device, num_queries, aux_loss, masks, panoptic, frozen_weights, bbox_loss_coef, giou_loss_coef, dice_loss_coef, mask_loss_coef, eos_coef, lr_backbone, set_cost_class, set_cost_bbox, set_cost_giou)
