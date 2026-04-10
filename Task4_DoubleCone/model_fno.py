import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FourierEncoding(nn.Module):
    """Fourier feature encoding for coordinates"""
    def __init__(self, in_dim=2, num_freqs=10):
        super().__init__()
        self.num_freqs = num_freqs
        self.out_dim = in_dim * 2 * num_freqs

    def forward(self, coords):
        out = []
        for i in range(self.num_freqs):
            freq = (2.0 ** i) * math.pi
            out.append(torch.sin(freq * coords))
            out.append(torch.cos(freq * coords))
        return torch.cat(out, dim=1)

class SpectralConv2d(nn.Module):
    """Spectral convolution layer for FNO"""
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, 
                            dtype=torch.cfloat, device=x.device)
        
        actual_modes1 = min(self.modes1, x_ft.size(-2))
        actual_modes2 = min(self.modes2, x_ft.size(-1))
        
        out_ft[:, :, :actual_modes1, :actual_modes2] = \
            self.compl_mul2d(x_ft[:, :, :actual_modes1, :actual_modes2], 
                           self.weights1[:, :, :actual_modes1, :actual_modes2])
        out_ft[:, :, -actual_modes1:, :actual_modes2] = \
            self.compl_mul2d(x_ft[:, :, -actual_modes1:, :actual_modes2], 
                           self.weights2[:, :, :actual_modes1, :actual_modes2])
        
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    """Fourier Neural Operator for Double Cone flow prediction"""
    def __init__(self, modes1=8, modes2=64, width=64, in_channels=5, out_channels=4, 
                 dropout=0.0, padding=0, use_fourier=True):
        super().__init__()
        
        self.use_fourier = use_fourier
        self.padding = padding
        
        if use_fourier:
            # Fourier feature encoding
            self.fourier_encoder = FourierEncoding(in_dim=2, num_freqs=10)
            self.augmented_in_channels = 3 + self.fourier_encoder.out_dim  # 3 + 40 = 43
        else:
            self.fourier_encoder = None
            self.augmented_in_channels = in_channels  # 5
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        
        # Lifting layer
        self.p = nn.Linear(self.augmented_in_channels, self.width)
        
        # Spectral convolution layers
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        # Bypass layers
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        
        # Projection layers
        self.q = nn.Linear(self.width, 128)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(128, out_channels)

    def forward(self, x):
        # x: [B, 5, 17, 384]
        if self.use_fourier:
            coords = x[:, 0:2, :, :]  # [B, 2, 17, 384]
            physics = x[:, 2:5, :, :]  # [B, 3, 17, 384]
            
            # Inject high-frequency features
            coords_feat = self.fourier_encoder(coords)  # [B, 40, 17, 384]
            x = torch.cat([physics, coords_feat], dim=1)  # [B, 43, 17, 384]
        
        # FNO processing
        x = x.permute(0, 2, 3, 1)  # [B, 17, 384, C]
        x = self.p(x)  # [B, 17, 384, width]
        x = x.permute(0, 3, 1, 2)  # [B, width, 17, 384]
        
        if self.padding > 0:
            x = F.pad(x, (0, self.padding, 0, self.padding))
        
        # Fourier layers
        for conv, w in zip([self.conv0, self.conv1, self.conv2, self.conv3], 
                          [self.w0, self.w1, self.w2, self.w3]):
            x1 = conv(x)
            x2 = w(x)
            x = x1 + x2
            x = F.gelu(x)
        
        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]
        
        # Projection
        x = x.permute(0, 2, 3, 1)  # [B, 17, 384, width]
        x = self.q(x)
        x = F.gelu(x)
        x = self.dropout_layer(x)
        x = self.fc(x)
        x = x.permute(0, 3, 1, 2)  # [B, 4, 17, 384]
        
        return x
