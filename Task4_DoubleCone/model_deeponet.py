import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """ Fourier Features for Coordinate Encoding """
    def __init__(self, in_dim=2, num_freqs=10):
        super().__init__()
        self.num_freqs = num_freqs
        self.out_dim = in_dim * (1 + 2 * num_freqs)

    def forward(self, x):
        out = [x]
        for i in range(self.num_freqs):
            freq = (2.0 ** i) * math.pi
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=-1)

class DenseNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_width=2048, hidden_depth=3):
        super().__init__()
        layers =[]
        layers.append(nn.Linear(input_dim, hidden_width))
        layers.append(nn.LayerNorm(hidden_width)) 
        layers.append(nn.GELU())
        for _ in range(hidden_depth):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(nn.LayerNorm(hidden_width)) 
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_width, output_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class DeepONet2d(nn.Module):
    def __init__(self, in_channels=5, out_channels=4, basis_size=256, use_fourier=False):
        super().__init__()
        self.basis_size = basis_size
        self.out_channels = out_channels
        self.use_fourier = use_fourier
        
        # Branch Net: Physical parameter inputs
        self.branch_net = DenseNet(input_dim=3, output_dim=out_channels * basis_size, 
                                   hidden_width=2048, hidden_depth=3)
        
        # Trunk Net: Coordinate inputs
        if self.use_fourier:
            self.pe = PositionalEncoding(in_dim=2, num_freqs=10)
            trunk_in_dim = self.pe.out_dim
        else:
            trunk_in_dim = 2
            
        self.trunk_net = DenseNet(input_dim=trunk_in_dim, output_dim=basis_size, 
                                  hidden_width=2048, hidden_depth=3)
        
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        B, C, H, W = x.shape
        branch_in = x[:, 2:5, 0, 0] # [B, 3]
        coords = x[0, 0:2, :, :].permute(1, 2, 0).reshape(-1, 2)
        
        if self.use_fourier:
            coords = self.pe(coords)
            
        branch_out = self.branch_net(branch_in)
        branch_out = branch_out.view(B, self.out_channels, self.basis_size)
        
        trunk_out = self.trunk_net(coords) 
        
        out = torch.matmul(branch_out, trunk_out.T)
        out = out + self.bias.view(1, -1, 1)
        
        return out.view(B, self.out_channels, H, W)
