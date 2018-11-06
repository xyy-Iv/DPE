import torch
import torch.nn as nn
import torch.nn.functional as F

class selu(nn.Module):
    def __init__(self):
        super(selu, self).__init__()

    def forward(self, x):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        
        return scale * F.elu(x, alpha)

class inclass(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inclass, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 5, stride=1, padding=2),
            selu(),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        x = self.conv(x)
        print(x.shape)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 5, stride=2, padding=2),
            selu(),
            nn.BatchNorm2d(out_ch)
            # nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.mpconv(x)
        print(x.shape)
        return x

class pool(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(pool, self).__init__()
        self.poolconv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 5, stride=2, padding=2),
            # nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        x = self.poolconv(x)
        print(x.shape)
        return x

class fc_expand(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(fc_expand, self).__init__()
        self.fceconv = nn.Sequential(
            nn.Linear(in_ch, out_ch, bias=False),
            selu()
        )
    
    def forward(self, x, ex):
        x = x.view(-1)
        x = self.fceconv(x)
        x = x.expand([ex, ex, 128]).permute(2, 1, 0).resize(1, 128, ex, ex)
        print(x.shape)
        return x

class mid_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(mid_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 5, padding=2)
        )

    def forward(self, x):
        x = self.conv(x)
        print(x.shape)
        return x


# class concat(nn.Module):
#     def __init__(self):
#         super(concat, self).__init__()
        
def concat( x1, x2):
    print(x1.shape, x2.shape)
    x = torch.cat([x1, x2], dim=1)
    return x

class resize(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(resize, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
    
    def forward(self, x):
        x = self.conv(x)
        print(x.shape)
        return x

class out_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(out_conv, self).__init__()
        self.conv = nn.Sequential(
            selu(),
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, 5, stride=2)
        )

    def forward(self, x):
        x = self.conv(x)
        print(x.shape)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.conv = out_conv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = resize.conv(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x


class outclass(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outclass, self).__init__()
        self.conv = out_conv(in_ch, out_ch)
    
    def forward(self, x1, x2):
        x = self.conv(x1)
        x = torch.add(x, x2)
        print(x.shape)
        return x