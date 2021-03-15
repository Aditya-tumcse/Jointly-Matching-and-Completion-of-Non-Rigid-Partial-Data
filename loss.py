import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self,margin = 2.0):
        super(ContrastiveLoss,self).__init__()
        self.margin = margin
        self.eps = 1e-9
    
    def forward(self,distances,target,size_average = True):
        losses = 0.5*((target.float() * distances) + (1 - target).float() * nn.functional.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        if size_average:
            return losses.mean()
        else:
            return losses.sum()