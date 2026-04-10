import torch
import torch.nn as nn

class BoltzmannDeepONet(nn.Module):
    def __init__(self, branch_dim=1, trunk_dim=2, hidden_dim=280, num_outputs=10, depth=5, activation='GELU'):
        """
        DeepONet for Cavity Task (Square Domain)
        Args:
            branch_dim: 1
            trunk_dim: 2 (x, y)
            hidden_dim: 280 (Target ~1.0M Params)
            num_outputs: 10 (Physics quantities)
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
        B_out = self.branch_net(x_branch)
        
        T_out = self.trunk_net(x_trunk)
        
        B_out_reshaped = B_out.view(-1, self.num_outputs, self.hidden_dim)
        
        # Dot product fusion
        prediction = torch.einsum("bkh, nh -> bkn", B_out_reshaped, T_out)
        
        prediction = prediction.permute(0, 2, 1)
        
        return prediction
