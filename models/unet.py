import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1,[diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))

class FluidUNet(nn.Module):
    """ Unified U-Net supporting arbitrary input resolutions and channels. """
    def __init__(self, in_channels=4, out_channels=4, base_dim=20, bilinear=True):
        super(FluidUNet, self).__init__()
        factor = 2 if bilinear else 1
        
        self.inc = DoubleConv(in_channels, base_dim)
        self.down1 = Down(base_dim, base_dim * 2)
        self.down2 = Down(base_dim * 2, base_dim * 4)
        self.down3 = Down(base_dim * 4, base_dim * 8)
        self.down4 = Down(base_dim * 8, base_dim * 16 // factor)
        
        self.up1 = Up(base_dim * 16 // factor + base_dim * 8, base_dim * 4 // factor, bilinear)
        self.up2 = Up(base_dim * 4 // factor + base_dim * 4, base_dim * 2 // factor, bilinear)
        self.up3 = Up(base_dim * 2 // factor + base_dim * 2, base_dim, bilinear)
        self.up4 = Up(base_dim + base_dim, base_dim, bilinear)
        
        self.outc = nn.Conv2d(base_dim, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)