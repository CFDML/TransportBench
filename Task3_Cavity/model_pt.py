import torch
import torch.nn as nn
import torch.nn.functional as F

class PointTransformer(nn.Module):
    def __init__(self, in_channels=3, out_channels=10, 
                 embed_dim=144, depth=4, num_heads=4, mlp_ratio=4., dropout=0.0):
        """
        Point Transformer for Cavity Flow
        Treats the 50x50 grid as a sequence of 2500 points.
        Target Params: ~1.0M
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Point Embedding
        self.embedding = nn.Linear(in_channels, embed_dim)
        
        # Positional Encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, 2500, embed_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Decoder
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, out_channels)

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
        B, H, W, C = x.shape
        
        x = x.view(B, -1, C)
        
        x = self.embedding(x)
        x = x + self.pos_embed 
        
        x = self.transformer(x)
        x = self.norm(x)
        
        x = self.head(x) 
        
        x = x.view(B, H, W, -1)
        
        return x