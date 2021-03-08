import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self,margin = 2.0):
        super(ContrastiveLoss,self).__init__()
        self.margin = margin

    def forward(self,out_1,out_2,label):
        euclidean_distance = nn.functional.pairwise_distance(out_1,out_2,keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +(label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive