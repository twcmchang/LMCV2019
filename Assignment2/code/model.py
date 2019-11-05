import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math

# Downsampling block with residual connection
# 2 layers
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [
            nn.Conv2d(in_size, out_size, 4, 2, 1),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_size, out_size, 1, 1, bias=False),
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.1))
        if dropout:
            layers.append(nn.Dropout(dropout))
        side = [
            nn.Conv2d(in_size, out_size, 2, 2, bias=False),
        ]
        self.model = nn.Sequential(*layers)
        self.side = nn.Sequential(*side)

    def forward(self, x):
        x = self.model(x) + self.side(x)
        return x

# Conv block with residual
class ConvBlock(nn.Module):
    def __init__(self, channel, normalize=True, dropout=0.0):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channel, channel, 1, 1, bias=False),
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(channel))
        layers.append(nn.LeakyReLU(0.1))
        if dropout:
            layers.append(nn.Dropout(dropout))
            
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x) + x
        return x
 
# FCN
class FCN(nn.Module):
    def __init__(self, in_channels=256, n_layer=2):
        super(FCN, self).__init__()

        def conv_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        layers = []
        for i in range(n_layer):
            layers.append(conv_block(in_channels, in_channels*2))
            in_channels = in_channels*2
        layers = [item for sub in layers for item in sub]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
  
class Dense(nn.Module):
    def __init__(self, in_channels=512 * 4, num_classes=100):
        super(Dense, self).__init__()

        self.model = nn.Sequential(        
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x_1, x_2, x_3, x_4):
        input = torch.cat((x_1, x_2, x_3, x_4), 1)
#         print(input.shape)
        return self.model(input)
    

class MultiFCN(nn.Module):
    def __init__(self, in_channels=3, num_classes=100):
        super(MultiFCN, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.conv1 = ConvBlock(64)
        self.down2 = UNetDown(64, 128)
        self.conv2 = ConvBlock(128)
        self.down3 = UNetDown(128, 256)
        self.conv3 = ConvBlock(256)
        self.down4 = UNetDown(256, 512)
        self.conv4 = ConvBlock(512)

        self.fcn1 = FCN(64, 3)
        self.fcn2 = FCN(128, 2)
        self.fcn3 = FCN(256, 1)
        
        self.down5 = UNetDown(512*4, 512)
        self.down6 = UNetDown(512, 512)
        self.down7 = UNetDown(512, 512)
        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool =  nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*4, num_classes)

#         self.dense = Dense(512*4,  num_classes)

    def forward(self, x):
        d1 = self.down1(x) # 64
        d2 = self.down2(d1) # 32
        d3 = self.down3(d2) # 16
        d4 = self.down4(d3) # 8
        
        
        c1 = self.conv1(d1)
        c2 = self.conv2(d2)
        c3 = self.conv3(d3)
        c4 = self.conv4(d4)
        
        f1 = self.fcn1(c1)
        f2 = self.fcn2(c2)
        f3 = self.fcn3(c3)
        f4 = c4

        f = torch.cat((f1, f2, f3, f4), 1)
        f = self.avgpool(f)
#         print(f.shape)
        
        f = f.view(f.size(0), -1)
#         print(x.shape)
        f = self.fc(f)
        

        return f