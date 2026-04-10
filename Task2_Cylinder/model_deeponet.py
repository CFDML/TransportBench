import torch
import torch.nn as nn

class BoltzmannDeepONet(nn.Module):
    def __init__(self, branch_dim, trunk_dim, hidden_dim=256, num_outputs=4, depth=5, activation='GELU'):
        """
        DeepONet Baseline Model (Standard 1M Params)
        """
        super().__init__()
        
        if activation == 'GELU':
            self.act = nn.GELU()
        elif activation == 'Tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Branch Net
        branch_layers = []
        branch_layers.extend([nn.Linear(branch_dim, hidden_dim), self.act])
        for _ in range(depth - 1):
            branch_layers.extend([nn.Linear(hidden_dim, hidden_dim), self.act])
        branch_layers.append(nn.Linear(hidden_dim, hidden_dim * num_outputs))
        self.branch_net = nn.Sequential(*branch_layers)

        # Trunk Net
        trunk_layers = []
        trunk_layers.extend([nn.Linear(trunk_dim, hidden_dim), self.act])
        for _ in range(depth - 1):
            trunk_layers.extend([nn.Linear(hidden_dim, hidden_dim), self.act])
        trunk_layers.extend([nn.Linear(hidden_dim, hidden_dim), self.act])
        self.trunk_net = nn.Sequential(*trunk_layers)
        
        self.num_outputs = num_outputs
        self.hidden_dim = hidden_dim

    def forward(self, x_branch, x_trunk):
        # B_out: [Batch, hidden * num_outputs]
        B_out = self.branch_net(x_branch)
        # T_out: [N_points, hidden]
        T_out = self.trunk_net(x_trunk)
        
        B_out_reshaped = B_out.view(-1, self.num_outputs, self.hidden_dim)
        
        # Dot Product: "bkh, nh -> bkn"
        prediction = torch.einsum("bkh, nh -> bkn", B_out_reshaped, T_out)
        prediction = prediction.permute(0, 2, 1)
        
        return prediction