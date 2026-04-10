import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2), # Downsampling
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            # Upsampling: output_padding=1 ensures the spatial dimensions are strictly doubled
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)

class AutoEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, base_dim=24):
        """
        Convolutional Autoencoder (CAE)
        Args:
            base_dim: 24 (Designed to hit ~1.0M Params)
        """
        super(AutoEncoder, self).__init__()
        
        # --- Encoder (Compression) ---
        # Input: [B, 3, 128, 128]
        self.enc1 = EncoderBlock(in_channels, base_dim)       # -> [B, 24, 64, 64]
        self.enc2 = EncoderBlock(base_dim, base_dim*2)        # -> [B, 48, 32, 32]
        self.enc3 = EncoderBlock(base_dim*2, base_dim*4)      # -> [B, 96, 16, 16]
        self.enc4 = EncoderBlock(base_dim*4, base_dim*8)      # -> [B, 192, 8, 8] (Latent)
        
        # --- Decoder (Reconstruction) ---
        # Note: No skip connections; direct reconstruction from the Latent space
        self.dec1 = DecoderBlock(base_dim*8, base_dim*4)      # -> [B, 96, 16, 16]
        self.dec2 = DecoderBlock(base_dim*4, base_dim*2)      # -> [B, 48, 32, 32]
        self.dec3 = DecoderBlock(base_dim*2, base_dim)        # -> [B, 24, 64, 64]
        
        # Final layer to restore the original resolution
        self.final_up = nn.ConvTranspose2d(base_dim, base_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.out_conv = nn.Conv2d(base_dim, out_channels, kernel_size=1) # -> [B, 4, 128, 128]

    def forward(self, x):
        # Encoder Path
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x) # Latent Representation
        
        # Decoder Path
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        
        x = self.final_up(x)
        out = self.out_conv(x)
        
        return out