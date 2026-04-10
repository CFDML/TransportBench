import torch
import torch.nn as nn

class AutoEncoder(nn.Module): 
    def __init__(self, in_channels=3, out_channels=10, base_dim=32): 
        super(AutoEncoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_dim, base_dim*2, kernel_size=3, stride=2, padding=1), # 25x25
            nn.ReLU(),
            nn.Conv2d(base_dim*2, base_dim*4, kernel_size=3, stride=2, padding=1), # 13x13
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_dim*4, base_dim*2, kernel_size=3, stride=2, padding=1, output_padding=0), # 25x25
            nn.ReLU(),
            nn.ConvTranspose2d(base_dim*2, base_dim, kernel_size=3, stride=2, padding=1, output_padding=1), # 50x50
            nn.ReLU(),
            nn.Conv2d(base_dim, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.permute(0, 2, 3, 1)
