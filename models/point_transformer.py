import torch
import torch.nn as nn

class PointTransformer(nn.Module):
    """
    Unified Point Transformer for sequence modeling of unstructured nodes.
    Removes hardcoded sequence lengths by mapping positional info via Linear layers.
    """
    def __init__(self, in_dim=4, out_dim=4, embed_dim=144, depth=4, num_heads=4, mlp_ratio=4., dropout=0.0):
        super().__init__()
        
        # Point Embedding + Dynamic Positional Encoding
        self.feat_emb = nn.Linear(in_dim, embed_dim)
        # Use coordinates (assumed first 2/3 dims) to generate positional features dynamically
        self.pos_emb = nn.Linear(in_dim, embed_dim) 
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Linear(embed_dim // 2, out_dim)
        )

    def forward(self, x):
        """ x:[Batch, N_points, In_dim] """
        # Reshape to sequence if input is a grid[B, C, H, W]
        is_grid = (x.dim() == 4)
        if is_grid:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
            
        feat = self.feat_emb(x)
        pos = self.pos_emb(x)
        x = feat + pos
        
        x = self.transformer(x)
        x = self.head(self.norm(x))
        
        if is_grid:
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
            
        return x