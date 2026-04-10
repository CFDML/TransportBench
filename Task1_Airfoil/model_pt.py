import torch
import torch.nn as nn

class PointTransformerBranch(nn.Module):
    def __init__(self, in_dim=2, d_model=144, nhead=4, num_layers=4, output_dim=256):
        super().__init__()
        
        # 1. Point Embedding (Coordinate -> Feature)
        # Map 2D coordinates to high dimensions, serving as both features and positional embeddings
        self.embedding = nn.Linear(in_dim, d_model)
        
        # 2. Positional Encoding
        # In Point Transformer, MLP mapping of coordinates is typically used directly as positional encoding
        self.pos_proj = nn.Linear(in_dim, d_model)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*2, # Reduce FFN ratio to control the number of parameters
            activation='gelu',
            batch_first=True,
            norm_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Output Projection
        self.fc_out = nn.Linear(d_model, output_dim)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, 337, 2]
        
        # Embedding & Positional Encoding
        feat = self.embedding(x)      # [B, N, D]
        pos = self.pos_proj(x)        # [B, N, D]
        x = feat + pos
        
        # Global Attention
        x = self.transformer(x)       # [B, N, D]
        
        # Global Pooling: Aggregate the point cloud sequence into a global geometry vector
        x = x.mean(dim=1)             # [B, D]
        
        x = self.norm(x)
        out = self.fc_out(x)          # [B, Output_Dim]
        return out

class PointTransformerONet(nn.Module):
    def __init__(self, hidden_dim=256, num_outputs=4):
        super().__init__()
        
        self.num_outputs = num_outputs
        self.hidden_dim = hidden_dim
        
        # --- Branch Net (PT) ---
        # Param check: 4 layers * 12 * 144^2 ~= 1.0M
        self.branch_net = PointTransformerBranch(
            in_dim=2, 
            d_model=144, 
            nhead=4, 
            num_layers=4, 
            output_dim=hidden_dim * num_outputs
        )

        # --- Trunk Net (MLP) ---
        # Consistent with DeepONet, processing coordinates
        self.act = nn.GELU()
        self.trunk_net = nn.Sequential(
            nn.Linear(2, hidden_dim), self.act,
            nn.Linear(hidden_dim, hidden_dim), self.act,
            nn.Linear(hidden_dim, hidden_dim), self.act,
            nn.Linear(hidden_dim, hidden_dim), self.act,
            nn.Linear(hidden_dim, hidden_dim), self.act
        )

    def forward(self, x_branch, x_trunk):
        # 1. Restore point cloud structure: [B, 674] -> [B, 337, 2]
        B = x_branch.shape[0]
        x_branch_points = x_branch.view(B, -1, 2)
        
        # 2. Branch Output: [B, hidden * outputs]
        B_out = self.branch_net(x_branch_points)
        
        # 3. Trunk Output: [N_grid, hidden]
        T_out = self.trunk_net(x_trunk)
        
        # 4. Dot Product Fusion
        B_out_reshaped = B_out.view(B, self.num_outputs, self.hidden_dim)
        
        # "bkh, nh -> bkn"
        prediction = torch.einsum("bkh, nh -> bkn", B_out_reshaped, T_out)
        
        # [B, N_grid, 4]
        prediction = prediction.permute(0, 2, 1)
        
        return prediction