import torch.nn as nn
import torch
import torch.nn.functional as F


class MaskMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask):
        pred = pred[mask == 1]
        target = target[mask == 1]
        return F.mse_loss(pred, target)



