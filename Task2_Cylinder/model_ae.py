import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, base_width=32):
        """
        Convolutional Autoencoder for Cylinder Flow
        Target Params: ~1M
        Input: (B, 4, 128, 192)
        Output: (B, 4, 128, 192)
        """
        super(Autoencoder, self).__init__()
        
        # --- Encoder (Downsampling) ---
        # 128x192 -> 64x96
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_width),
            nn.GELU()
        )
        # 64x96 -> 32x48
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_width, base_width*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.GELU()
        )
        # 32x48 -> 16x24
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_width*2, base_width*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.GELU()
        )
        # 16x24 -> 8x12 (Bottleneck)
        self.enc4 = nn.Sequential(
            nn.Conv2d(base_width*4, base_width*8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.GELU()
        )

        # --- Decoder (Upsampling) ---
        # 8x12 -> 16x24
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_width*8, base_width*4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.GELU()
        )
        # 16x24 -> 32x48
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_width*4, base_width*2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.GELU()
        )
        # 32x48 -> 64x96
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_width*2, base_width, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_width),
            nn.GELU()
        )
        # 64x96 -> 128x192
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(base_width, base_width, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_width),
            nn.GELU()
        )
        
        # Output Head
        self.final_conv = nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (B, 4, 128, 192)
        
        # Encoder
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x) # Latent representation
        
        # Decoder
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        
        out = self.final_conv(x)
        return out
