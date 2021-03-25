import torch.nn as nn
import torch
from torchsummary import summary

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,out_planes,kernel_size=3,stride=stride,padding=1,bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,out_planes,kernel_size=1,stride=stride,bias=False)

class ResBlock(nn.Module):
    def __init__(self,in_channels,planes,stride=1):
        super(ResBlock,self).__init__()
        self.conv1 = conv3x3x3(in_channels,planes,stride=stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU()
        #self.conv2 = conv3x3x3(planes,planes,stride=stride)
        #self.bn2 = nn.BatchNorm3d(planes)
        self.stride = stride
        self.conv1x1x1 = conv1x1x1(in_channels,planes,stride=1)

    def forward(self,x):
        residual = self.conv1x1x1(x)

        out = self.relu(self.bn1(self.conv1(x)))
        #out = self.bn2(self.conv2(out))

        #Adding skip connection
        out += residual
        out = self.relu(out)

        return out

class HourGlassNet(nn.Module):
    def __init__(self,in_channels=1):
        super(HourGlassNet,self).__init__()

        #Encoder
        self.e1 = nn.Sequential(nn.Conv3d(in_channels,64,kernel_size=(7,7,7),stride=(2,2,2),padding=(3,3,3)),ResBlock(64,64)) #8x8x8x64
        self.e1add = nn.Sequential(conv3x3x3(64,64),nn.BatchNorm3d(64),nn.ReLU(inplace=True))

        self.e2 = nn.Sequential(nn.Conv3d(64,128,kernel_size=(4,4,4),stride=(2,2,2),padding=(1,1,1),bias=False),nn.BatchNorm3d(128),ResBlock(128,128)) #4x4x4x128
        self.e2add = nn.Sequential(conv3x3x3(128,128,stride=1),nn.BatchNorm3d(128),nn.ReLU(inplace=True)) #4x4x4x128

        self.e3 = nn.Sequential(nn.Conv3d(128,256,kernel_size=(4,4,4),stride=(2,2,2),padding=(1,1,1),bias=False),nn.BatchNorm3d(256),ResBlock(256,256))#2x2x2x256
        self.e3add = nn.Sequential(conv3x3x3(256,256,stride=1),nn.BatchNorm3d(256),nn.ReLU(inplace=True))#2x2x2x256

        self.e4 = nn.Sequential(nn.Conv3d(256,512,kernel_size=(4,4,4),stride=(2,2,2),padding=(1,1,1),bias=False),nn.BatchNorm3d(512),ResBlock(512,512)) #1x1x1x512
        
        #Decoder
        self.d1 = nn.Sequential(nn.Upsample(scale_factor=2.0,mode='nearest'),conv1x1x1(512,256))#2x2x2x256

        self.d2 = nn.Sequential(conv1x1x1(512,256),ResBlock(256,256),nn.BatchNorm3d(256)) #2x2x2x256

        self.d3 = nn.Sequential(nn.Upsample(scale_factor=2.0,mode='nearest'),conv1x1x1(256,128)) #4x4x4x128
        self.d4 = nn.Sequential(conv1x1x1(256,128),ResBlock(128,128),nn.BatchNorm3d(128)) #4x4x4x128

        self.d5 = nn.Sequential(nn.Upsample(scale_factor=2.0,mode='nearest'),conv1x1x1(128,64)) #8x8x8x64
        self.d6 = nn.Sequential(conv1x1x1(128,64),ResBlock(64,64),nn.BatchNorm3d(64)) #8x8x8x64

        self.d7 = nn.Upsample(scale_factor=2.0,mode='nearest') #16x16x16x64
        self.final_out = nn.Sequential(nn.Conv3d(64,1,kernel_size=(4,4,4),stride=(1,1,1),padding=(1,1,1)),nn.ReLU(inplace=True))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        en1 = self.e1(x)
        en1add = self.e1add(en1)
        
        en2 = self.e2(en1add)
        en2add = self.e2add(en2)
        
        en3 = self.e3(en2add)
        en3add = self.e3add(en3)
        
        en4 = self.e4(en3add)
        
        
        dn1 = self.d1(en4)
        dn1conc = torch.cat([dn1,en3add],1)
        dn1r = self.d2(dn1conc)
        
        dn2 = self.d3(dn1r)
        dn2conc = torch.cat([dn2,en2add],1)
        dn2r = self.d4(dn2conc)
        

        dn3 = self.d5(dn2r)
        dn3conc = torch.cat([dn3,en1add],1)
        print(dn3conc.shape)
        dn3r = self.d6(dn3conc)
        

        dn4 = self.d7(dn3r)
        final_output = self.final_out(dn4)

        return final_output
    
if __name__ == '__main__':
    net = HourGlassNet()
    summary(net,(1,15,15,15))
