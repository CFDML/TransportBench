import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, img_size=(128, 192), patch_size=8, in_chans=4, out_chans=4, 
                 embed_dim=144, depth=4, num_heads=4, mlp_ratio=4., p_dropout=0.):
        """
        ViT for Cylinder Flow (Dense Regression)
        Target Params: ~1.1M (with embed_dim=144, depth=4)
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_h = img_size[0] // patch_size # 16
        self.grid_w = img_size[1] // patch_size # 24
        self.num_patches = self.grid_h * self.grid_w
        self.embed_dim = embed_dim
        self.out_chans = out_chans

        # 1. Patch Embedding
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=p_dropout)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                                   dim_feedforward=int(embed_dim * mlp_ratio), 
                                                   dropout=p_dropout, activation='gelu', 
                                                   batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        # 4. Decoder Head
        self.decoder_pred = nn.Linear(embed_dim, patch_size**2 * out_chans)

    def forward(self, x):
        # x: (B, 4, 128, 192)
        B, C, H, W = x.shape
        
        # Embed: (B, Embed, Gh, Gw) -> (B, Np, Embed)
        x = self.proj(x).flatten(2).transpose(1, 2)
        
        # Add Pos Embed
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer Blocks
        x = self.blocks(x)
        x = self.norm(x)
        
        # Decode
        x = self.decoder_pred(x)
        
        # Reshape back to image
        x = x.view(B, self.grid_h, self.grid_w, self.patch_size, self.patch_size, self.out_chans)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, self.out_chans, H, W)
        
        return x
