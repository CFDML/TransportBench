import torch
import torch.nn as nn

class PointTransformer(nn.Module):
    def __init__(self, in_dim=4, out_dim=4, embed_dim=144, depth=4, num_heads=4, mlp_ratio=4., dropout=0.0):
        """
        Configured for ~1M parameters match:
        - embed_dim: 144
        - depth: 4
        """
        super().__init__()
        
        self.embedding = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
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
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, out_dim)
        )

    def forward(self, x):
        x = self.embedding(x) 
        x = self.transformer(x)
        x = self.norm(x)
        x = self.head(x)
        return x
