import torch
import torch.nn as nn


class Siamese(nn.Module):
    def __init__(self,model):
        super(Siamese,self).__init__()
        self.model = model


    def forward(self,input_1,input_2):
        out_1 = self.model(input_1)
        out_2 = self.model(input_2)
        return(out_1,out_2)

    def _get_model_(self,x):
        return(self.model(x))

"""
#for test.Import 3D ResNet architecture to test
if __name__ == '__main__':
    net = Siamese(resnet.generate_model(10))
    print(net)
    print(list(net.parameters()))
"""