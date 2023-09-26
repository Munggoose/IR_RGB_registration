
import torch
import torch.nn as nn
import wandb
import numpy as np

def calcualte_IOU(pred, label):
    pred_bool = pred.to(torch.bool)
    label_bool = label.to(torch.bool)
    overlap_area = torch.logical_and(pred_bool, label_bool)
    combine_area = torch.logical_or(pred_bool, label_bool)
    
    iou = torch.sum(overlap_area).item() / torch.sum(combine_area).item()
    
    return iou