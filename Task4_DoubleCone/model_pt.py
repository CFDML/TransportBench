import torch
import torch.nn as nn
import math

class FourierEncoding(nn.Module):
    """Fourier feature encoding for coordinates"""
    def __init__(self, in_dim=2, num_freqs=4):
        super().__init__()
        self.num_freqs = num_freqs
        self.out_dim = in_dim * 2 * num_freqs

    def forward(self, coords):
        out = []
        for i in range(self.num_freqs):
            freq = (2.0 ** i) * math.pi
            out.append(torch.sin(freq * coords))
            out.append(torch.cos(freq * coords))
        return torch.cat(out, dim=-1)

class PointTransformer(nn.Module):
    """Point Latent Transformer for Double Cone flow prediction"""
    def __init__(self, in_channels=5, out_channels=4, 
                 latent_dim=512, num_latents=1024, 
                 depth=10, num_heads=8, mlp_ratio=4.0, use_fourier=True):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.out_channels = out_channels
        self.use_fourier = use_fourier

        if use_fourier:
            # Fourier encoding with reduced frequency
            self.fourier_encoder = FourierEncoding(in_dim=2, num_freqs=4)
            augmented_dim = 3 + self.fourier_encoder.out_dim  # 3 + 16 = 19
        else:
            self.fourier_encoder = None
            augmented_dim = in_channels  # 5

        # Point embedding
        self.point_embed = nn.Linear(augmented_dim, latent_dim)
        
        # Learnable latent variables
        self.latents = nn.Parameter(torch.randn(1, num_latents, latent_dim))

        # Encoder cross-attention
        self.enc_cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.enc_norm1 = nn.LayerNorm(latent_dim)
        self.enc_norm2 = nn.LayerNorm(latent_dim)
        
        # Latent processor
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, 
            nhead=num_heads, 
            dim_feedforward=int(latent_dim * mlp_ratio), 
            activation='gelu', 
            batch_first=True, 
            norm_first=True
        )
        self.processor = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Decoder cross-attention
        self.dec_cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.dec_norm1 = nn.LayerNorm(latent_dim)
        self.dec_norm2 = nn.LayerNorm(latent_dim)

        # Output head
        self.head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, out_channels)
        )

    def forward(self, x):
        # x: [B, 5, 17, 384]
        B, C, H, W = x.shape
        N = H * W
        
        # Reshape to point cloud
        points = x.view(B, C, N).transpose(1, 2)  # [B, N, 5]
        
        if self.use_fourier:
            coords = points[:, :, 0:2]  # [B, N, 2]
            physics = points[:, :, 2:5]  # [B, N, 3]
            
            # Fourier encoding
            coords_feat = self.fourier_encoder(coords)  # [B, N, 16]
            points_aug = torch.cat([physics, coords_feat], dim=-1)  # [B, N, 19]
        else:
            points_aug = points  # [B, N, 5]
        
        # Point embedding
        p_feat = self.point_embed(points_aug)  # [B, N, latent_dim]

        # Encoder: compress points to latents
        latents = self.latents.expand(B, -1, -1)  # [B, num_latents, latent_dim]
        latents_attn, _ = self.enc_cross_attn(
            query=self.enc_norm1(latents), 
            key=p_feat, 
            value=p_feat
        )
        latents = latents + latents_attn
        
        # Process latents
        latents = self.processor(latents)

        # Decoder: reconstruct from latents to points
        out_feat, _ = self.dec_cross_attn(
            query=self.dec_norm1(p_feat), 
            key=self.dec_norm2(latents), 
            value=self.dec_norm2(latents)
        )
        out_feat = p_feat + out_feat

        # Output prediction
        out = self.head(out_feat)  # [B, N, out_channels]
        out = out.transpose(1, 2).view(B, self.out_channels, H, W)
        
        return out