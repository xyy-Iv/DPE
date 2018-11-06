import torch
import torch.nn as nn
from parts import *

class DPENet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(DPENet, self).__init__()
        self.inc = inclass(n_channels, 16)
        self.down1 = down(16, 32)
        self.down2 = down(32, 64)
        self.down3 = down(64, 128)
        self.down4 = down(128, 128)
        self.pool = pool(128, 128)
        self.fc = fc_expand(8192, 128)
        self.mconv1 = mid_conv(128, 128)
        self.mconv2 = mid_conv(256, 128)
        self.resi = resize(128, 128)
        self.up1 = up(256, 128)
        self.up2 = up(192, 64)
        self.up3 = up(96, 32)
        self.out1 = out_conv(48, 16)
        self.out2 = out_conv(16, 3)
        self.outc = outclass(3, n_classes)

    def forward(self, x):
        x_clone = x.clone()
        print(x.shape)
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
        x12 = self.resi(x11)
        x = self.up1(x12, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.out1(x)
        x = self.out2(x)
        x = self.outc(x, x_clone)

net = DPENet(3, 3)
print(net)

input = torch.randn(1, 3, 512, 512)
out = net(input)
print(out)