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

class VisionTransformer(nn.Module):
    """Vision Transformer for Double Cone flow prediction"""
    def __init__(self, in_channels=5, out_channels=4, 
                 img_size=(17, 384), patch_size=(1, 8), 
                 embed_dim=512, depth=10, num_heads=8, mlp_ratio=4.0, use_fourier=True):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size
        self.pH, self.pW = patch_size
        self.use_fourier = use_fourier
        
        if use_fourier:
            # Fourier feature encoding
            self.fourier_encoder = FourierEncoding(in_dim=2, num_freqs=10)
            self.augmented_in_channels = 3 + self.fourier_encoder.out_dim  # 3 + 40 = 43
        else:
            self.fourier_encoder = None
            self.augmented_in_channels = in_channels  # 5
        
        self.num_patches = (self.H // self.pH) * (self.W // self.pW)
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(self.augmented_in_channels, embed_dim, 
                                     kernel_size=patch_size, stride=patch_size)
        
        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)
        
        # Transformer encoder blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=int(embed_dim * mlp_ratio), 
            activation='gelu', 
            batch_first=True, 
            norm_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Output head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, out_channels * self.pH * self.pW)

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
        
        # Patch embedding
        x = self.patch_embed(x_aug)  # [B, embed_dim, H//pH, W//pW]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add positional encoding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer processing
        x = self.blocks(x)
        x = self.norm(x)
        
        # Reconstruction
        x = self.head(x)  # [B, num_patches, out_channels * pH * pW]
        
        # Reshape back to image
        x = x.view(x.shape[0], self.H // self.pH, self.W // self.pW, -1, self.pH, self.pW)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(x.shape[0], -1, self.H, self.W)
        
        return x