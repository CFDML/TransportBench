import torch
import torch.nn as nn
import math

class FourierEncoding(nn.Module):
    """Fourier feature encoding for coordinates"""
    def __init__(self, in_dim=2, num_freqs=10):
        super().__init__()
        self.num_freqs = num_freqs
        self.out_dim = in_dim * 2 * num_freqs

    def forward(self, coords):
        out = []
        for i in range(self.num_freqs):
            freq = (2.0 ** i) * math.pi
            out.append(torch.sin(freq * coords))
            out.append(torch.cos(freq * coords))
        return torch.cat(out, dim=1)

class DoubleConv(nn.Module):
    """Double convolution block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
    
    def forward(self, x):
        return self.double_conv(x)

class FluidUNet(nn.Module):
    """U-Net architecture for Double Cone flow prediction"""
    def __init__(self, in_channels=5, out_channels=4, features=64, use_fourier=True):
        super().__init__()
        
        self.use_fourier = use_fourier
        
        if use_fourier:
            # Fourier encoding
            self.fourier_encoder = FourierEncoding(in_dim=2, num_freqs=10)
            augmented_in_channels = 3 + self.fourier_encoder.out_dim  # 3 + 40 = 43
        else:
            self.fourier_encoder = None
            augmented_in_channels = in_channels  # 5
        
        self.out_channels = out_channels
        
        # Encoder
        self.inc = DoubleConv(augmented_in_channels, features)
        self.down1 = nn.Sequential(nn.MaxPool2d((1, 2)), DoubleConv(features, features*2))
        self.down2 = nn.Sequential(nn.MaxPool2d((1, 2)), DoubleConv(features*2, features*4))
        self.down3 = nn.Sequential(nn.MaxPool2d((1, 2)), DoubleConv(features*4, features*8))
        self.down4 = nn.Sequential(nn.MaxPool2d((1, 2)), DoubleConv(features*8, features*16))
        
        # Decoder with skip connections
        self.up1 = nn.ConvTranspose2d(features*16, features*8, kernel_size=(1, 2), stride=(1, 2))
        self.conv1 = DoubleConv(features*16, features*8)
        
        self.up2 = nn.ConvTranspose2d(features*8, features*4, kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = DoubleConv(features*8, features*4)
        
        self.up3 = nn.ConvTranspose2d(features*4, features*2, kernel_size=(1, 2), stride=(1, 2))
        self.conv3 = DoubleConv(features*4, features*2)
        
        self.up4 = nn.ConvTranspose2d(features*2, features, kernel_size=(1, 2), stride=(1, 2))
        self.conv4 = DoubleConv(features*2, features)
        
        self.outc = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # x: [B, 5, 17, 384]
        if self.use_fourier:
            coords = x[:, 0:2, :, :]  # [B, 2, 17, 384]
            physics = x[:, 2:5, :, :]  # [B, 3, 17, 384]
            
            # Inject high-frequency coordinate features
            coords_feat = self.fourier_encoder(coords)  # [B, 40, 17, 384]
            x_aug = torch.cat([physics, coords_feat], dim=1)  # [B, 43, 17, 384]
        else:
            x_aug = x  # [B, 5, 17, 384]
        
        # Encoder path with skip connections
        x1 = self.inc(x_aug)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path
        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv4(x)
        
        return self.outc(x)
