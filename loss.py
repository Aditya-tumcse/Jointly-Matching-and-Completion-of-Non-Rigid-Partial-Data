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

class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = nn.functional.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()