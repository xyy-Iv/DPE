import torch
import torch.nn as nn
from parts import *

class DPENet_gen(nn.Module):
    def __init__(self, ngpu):
        super(DPENet_gen, self).__init__()
        self.ngpu = ngpu

        self.inc = inclass(3, 16)
        self.down1 = down(16, 32)
        self.down2 = down(32, 64)
        self.down3 = down(64, 128)
        self.down4 = down(128, 128)
        self.pool = pool(128, 128)
        self.fc = fc_expand(8192, 128)
        self.mconv1 = mid_conv(128, 128)
        self.mconv2 = mid_conv(256, 128)
        self.resi = resi(128, 128)
        self.up1 = up(256, 128)
        self.up2 = up(192, 64)
        self.up3 = up(96, 32)
        self.up4 = up(48, 16)
        self.out1 = out_conv(16, 3)
        self.outc = outclass(3, 3)

    def forward(self, x):
        x_clone = x.clone()
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x6 = self.pool(x5)
        x7 = self.pool(x6)
        x8 = self.fc(x7, 32)
        
        x9 = self.mconv1(x5)
        x10 = concat(x9, x8)
        x11 = self.mconv2(x10)
        
        x = self.up1(x11, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out1(x)
        x = self.outc(x, x_clone)
        #print(x.shape)

        return x

class DPENet_dis(nn.Module):
    def __init__(self, ngpu):
        super(DPENet_dis, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, 16, 5, padding=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(16),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(16, 32, 5, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(32, 64, 5, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(64),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(64, 128, 5, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(64),

            nn.Conv2d(128, 128, 5, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(128),

            nn.Conv2d(128, 128, 5, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(128),
            # # state size. (ndf*8) x 4 x 4
            out_dis(32768, 1)
            #nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

#net = DPENet_dis(1)
#print(net)
#input = torch.randn(2, 3, 512, 512)
#out = net(input)
#print(out)