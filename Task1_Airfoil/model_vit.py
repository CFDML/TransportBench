import torch
import torch.nn as nn
import numpy as np

class PatchEmbed(nn.Module):
    def __init__(self, img_size=(128, 128), patch_size=16, in_chans=3, embed_dim=144):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # Use convolution to implement patching and linear projection
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, D, Gh, Gw] -> [B, D, N] -> [B, N, D]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        # qkv: [B, N, 3*C] -> [B, N, 3, H, C//H] -> [3, B, H, N, C//H]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=3., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=(128, 128), patch_size=16, in_chans=3, out_chans=4, 
                 embed_dim=144, depth=4, num_heads=4, mlp_ratio=3.):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_chans = out_chans
        
        # 1. Patch Embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # 2. Positional Embedding (Learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # 3. Transformer Encoder Blocks
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # 4. Decoder (Linear Projection back to pixels)
        # Map each Token back to patch_size * patch_size * out_channels
        self.decoder_pred = nn.Linear(embed_dim, patch_size * patch_size * out_chans, bias=True)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # --- Encoder ---
        x = self.patch_embed(x) # [B, N, D]
        x = x + self.pos_embed  # Add Position Info
        x = self.blocks(x)
        x = self.norm(x)
        
        # --- Decoder (Reshape back to Image) ---
        x = self.decoder_pred(x) # [B, N, P*P*OutC]
        
        # Reshape logic: [B, Gh*Gw, P*P*C] -> [B, OutC, H, W]
        P = self.patch_size
        Gh, Gw = self.patch_embed.grid_size
        C_out = self.out_chans
        
        # [B, Gh, Gw, P, P, C_out]
        x = x.view(B, Gh, Gw, P, P, C_out)
        # Permute to [B, C_out, Gh, P, Gw, P] -> [B, C_out, H, W]
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C_out, H, W)
        
        return x