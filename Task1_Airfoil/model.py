import torch
import torch.nn as nn

class BoltzmannDeepONet(nn.Module):
    def __init__(self, branch_dim, trunk_dim, hidden_dim=128, num_outputs=4, depth=3, activation='GELU'):
        super().__init__()
        
        # 1. Select activation function
        if activation == 'GELU':
            self.act = nn.GELU()
        elif activation == 'Tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # 2. Dynamically build Branch Net
        # Structure: Input -> [Hidden -> Act] x depth -> Output
        branch_layers = []
        # First layer: Input layer -> Hidden layer
        branch_layers.extend([nn.Linear(branch_dim, hidden_dim), self.act])
        
        # Intermediate layers: Loop and add (depth - 1) times
        for _ in range(depth - 1):
            branch_layers.extend([nn.Linear(hidden_dim, hidden_dim), self.act])
            
        # Final layer: Linear projection to (hidden_dim * num_outputs)
        # Note: The final layer typically does not use an activation function to allow for linear combination
        branch_layers.append(nn.Linear(hidden_dim, hidden_dim * num_outputs))
        
        self.branch_net = nn.Sequential(*branch_layers)

        # 3. Dynamically build Trunk Net
        trunk_layers = []
        # First layer: Input layer -> Hidden layer
        trunk_layers.extend([nn.Linear(trunk_dim, hidden_dim), self.act])
        
        # Intermediate layers
        for _ in range(depth - 1):
            trunk_layers.extend([nn.Linear(hidden_dim, hidden_dim), self.act])
            
        # Final layer: Output hidden_dim (acts as basis functions)
        trunk_layers.extend([nn.Linear(hidden_dim, hidden_dim), self.act])
        
        self.trunk_net = nn.Sequential(*trunk_layers)
        
        # Save parameters for the Forward pass
        self.num_outputs = num_outputs
        self.hidden_dim = hidden_dim

    def forward(self, x_branch, x_trunk):
        """
        x_branch: [Batch, branch_dim]
        x_trunk:  [N_points, trunk_dim]
        """
        # B_out: [Batch, hidden * num_outputs]
        B_out = self.branch_net(x_branch)
        
        # T_out: [N_points, hidden]
        T_out = self.trunk_net(x_trunk)
        
        # Reshape Branch output: [Batch, num_outputs, hidden]
        B_out_reshaped = B_out.view(-1, self.num_outputs, self.hidden_dim)
        
        # Dot Product Fusion (Multi-output)
        # Einsum: Batch(b), Output(k), Hidden(h), Points(n)
        # "bkh, nh -> bkn"
        prediction = torch.einsum("bkh, nh -> bkn", B_out_reshaped, T_out)
        
        # Transpose to [Batch, N_points, num_outputs] to match the Label
        prediction = prediction.permute(0, 2, 1)
        
        return prediction