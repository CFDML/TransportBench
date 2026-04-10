import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    """ 
    Unified Convolutional AutoEncoder.
    Uses Bilinear Upsampling instead of ConvTranspose to avoid dimension mismatch 
    across different geometric shapes (50x50 vs 128x192).
    """
    def __init__(self, in_channels=4, out_channels=4, base_dim=32):
        super(AutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, 3, stride=2, padding=1), nn.BatchNorm2d(base_dim), nn.GELU(),
            nn.Conv2d(base_dim, base_dim*2, 3, stride=2, padding=1), nn.BatchNorm2d(base_dim*2), nn.GELU(),
            nn.Conv2d(base_dim*2, base_dim*4, 3, stride=2, padding=1), nn.BatchNorm2d(base_dim*4), nn.GELU(),
            nn.Conv2d(base_dim*4, base_dim*8, 3, stride=2, padding=1), nn.BatchNorm2d(base_dim*8), nn.GELU()
        )
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_dim*8, base_dim*4, 3, padding=1), nn.BatchNorm2d(base_dim*4), nn.GELU(),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_dim*4, base_dim*2, 3, padding=1), nn.BatchNorm2d(base_dim*2), nn.GELU(),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_dim*2, base_dim, 3, padding=1), nn.BatchNorm2d(base_dim), nn.GELU(),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_dim, base_dim, 3, padding=1), nn.BatchNorm2d(base_dim), nn.GELU()
        )
        self.final_conv = nn.Conv2d(base_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        target_size = x.shape[2:] # (H, W)
        x = self.encoder(x)
        x = self.decoder(x)
        # Interpolate to strictly match original size (crucial for odd dimensions like 50x50)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
        return self.final_conv(x)