import torch
import torch.nn as nn
import torch.nn.functional as F

class Siamese(nn.Module):
    def __init__(self,model):
        super(Siamese,self).__init__()
        self.model = model
        self.fc = nn.Linear(512,2)

    def forward_once(self,x):
        out = self.model(x)
        out = F.relu(self.fc(out))
        return out

    def forward(self,input_1,input_2):
        out_1 = self.forward_once(input_1)
        out_2 = self.forward_once(input_2)
        return(out_1,out_2)