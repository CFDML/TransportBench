import torch
import torch.nn as nn

class BoltzmannDeepONet(nn.Module):
    """
    Unified DeepONet for TransportBench.
    Supports dynamic branch and trunk input dimensions.
    """
    def __init__(self, branch_dim, trunk_dim, hidden_dim=256, num_outputs=4, depth=5, activation='GELU'):
        super().__init__()
        
        self.act = nn.GELU() if activation == 'GELU' else nn.Tanh()

        # --- Branch Net ---
        branch_layers = [nn.Linear(branch_dim, hidden_dim), self.act]
        for _ in range(depth - 1):
            branch_layers.extend([nn.Linear(hidden_dim, hidden_dim), self.act])
        branch_layers.append(nn.Linear(hidden_dim, hidden_dim * num_outputs))
        self.branch_net = nn.Sequential(*branch_layers)

        # --- Trunk Net ---
        trunk_layers =[nn.Linear(trunk_dim, hidden_dim), self.act]
        for _ in range(depth - 1):
            trunk_layers.extend([nn.Linear(hidden_dim, hidden_dim), self.act])
        trunk_layers.extend([nn.Linear(hidden_dim, hidden_dim), self.act])
        self.trunk_net = nn.Sequential(*trunk_layers)
        
        self.num_outputs = num_outputs
        self.hidden_dim = hidden_dim

    def forward(self, x_branch, x_trunk):
        """
        x_branch: [Batch, branch_dim]
        x_trunk:[N_points, trunk_dim]
        Output:   [Batch, N_points, num_outputs]
        """
        B_out = self.branch_net(x_branch)           # [B, hidden * outputs]
        T_out = self.trunk_net(x_trunk)             # [N_points, hidden]
        
        B_out_reshaped = B_out.view(-1, self.num_outputs, self.hidden_dim)
        
        # Dot Product Fusion: "bkh, nh -> bkn"
        prediction = torch.einsum("bkh, nh -> bkn", B_out_reshaped, T_out)
        prediction = prediction.permute(0, 2, 1)    # [B, N_points, num_outputs]
        
        return prediction