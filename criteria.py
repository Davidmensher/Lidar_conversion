import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target ,pred_normal, target_normal):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        l2_loss = (diff ** 2).mean()

        criterion = nn.CosineSimilarity(dim=0)
        cos_value = criterion(target_normal, pred_normal)
        cos_value_flatten = torch.flatten(cos_value)
        mean = torch.mean(cos_value_flatten)
        normal_loss = 1 - mean


        return 0.9* l2_loss + 0.1* normal_loss

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target,pred_normal, target_normal):
        
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        l1_loss = diff.abs().mean()


        #normal_loss = torch.mean(normal_loss)
        criterion = nn.CosineSimilarity(dim=0)
        cos_value = criterion(target_normal, pred_normal)
        cos_value_flatten = torch.flatten(cos_value)
        mean = torch.mean(cos_value_flatten)
        normal_loss = 1 - mean
        #end normal loss


        self.loss = 0.9* l1_loss + 0.1*normal_loss
        return self.loss

