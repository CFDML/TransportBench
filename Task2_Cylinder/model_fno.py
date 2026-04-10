import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Fourier modes to keep (H)
        self.modes2 = modes2 # Fourier modes to keep (W)

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Apply FFT to spatial dimensions
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        # Extract lower frequency modes
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1=12, modes2=12, width=28, in_channels=4, out_channels=4):
        """
        Cylinder FNO:
        in_channels = 4 (Kn_map, Ma_map, x, y)
        """
        super(FNO2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        
        # Lifting: 4 -> 28
        self.fc0 = nn.Linear(in_channels, width) 

        self.conv0 = SpectralConv2d(width, width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(width, width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(width, width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(width, width, self.modes1, self.modes2)
        
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        # x: [B, 4, H, W]
        x = x.permute(0, 2, 3, 1) # [B, H, W, 4]
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2) # [B, 28, H, W]

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2 

        x = x.permute(0, 2, 3, 1) # [B, H, W, 28]
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x) # [B, H, W, 4]
        
        # Output: [B, H, W, 4] -> Permute back to [B, 4, H, W]
        x = x.permute(0, 3, 1, 2)
        return x