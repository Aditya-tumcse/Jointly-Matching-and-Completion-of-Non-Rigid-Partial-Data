import torch
import torch.nn as nn
from torchsummary import summary

class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(nn.Conv3d(in_channels,out_channels,3,1,1,bias=False),nn.BatchNorm3d(out_channels),nn.ReLU(inplace=True),nn.Conv3d(out_channels,out_channels,3,1,1,bias=False),nn.BatchNorm3d(out_channels),nn.ReLU(inplace=True))

    def forward(self,x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self,in_channels=1,out_channels=1,features=[64, 128, 256]):
        super(UNet,self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2,stride=2)

        #Downpart of UNet
        for feature in features:
            self.downs.append(DoubleConv(in_channels,feature))
            in_channels = feature

        #Uppart of UNet
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose3d(feature * 2,feature,kernel_size=2,stride=2))
            self.ups.append(DoubleConv(feature * 2,feature))

        self.bottleneck = DoubleConv(features[-1],features[-1] * 2)
        self.final_conv = nn.Conv3d(features[0],out_channels,kernel_size=1)

    def forward(self,x):
        skip_connections = []

        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0,len(self.ups),2):
            #print(idx)
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            print(skip_connection.shape)
            concat_skip = torch.cat((skip_connection,x),dim=1)
            x = self.ups[idx + 1](concat_skip)
            
        return self.final_conv(x)
"""
#Testing
if __name__ == '__main__':
    net = UNet()
    summary(net,(1,15,15,15))
 """ 
