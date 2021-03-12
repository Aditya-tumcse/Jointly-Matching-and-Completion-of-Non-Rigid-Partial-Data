import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self,margin = 2.0):
        super(ContrastiveLoss,self).__init__()
        self.margin = margin
        self.eps = 1e-6
    
    def forward(self,distances,target,size_average = True):
        losses = 0.5*((target.float32() * distances) + (1 - target).float32() * nn.functional.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        if size_average:
            return losses.mean()
        else:
            return losses.sum()