import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=(128, 192), patch_size=8, in_chans=4, embed_dim=144):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.grid_h, self.grid_w = img_size[0] // patch_size, img_size[1] // patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class VisionTransformer(nn.Module):
    """ Unified ViT for Dense Regression across all TransportBench Tasks. """
    def __init__(self, img_size=(128, 192), patch_size=8, in_chans=4, out_chans=4, 
                 embed_dim=144, depth=4, num_heads=4, mlp_ratio=4., p_dropout=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.out_chans = out_chans
        self.grid_h = self.patch_embed.grid_h
        self.grid_w = self.patch_embed.grid_w
        self.patch_size = patch_size

        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                                   dim_feedforward=int(embed_dim * mlp_ratio), 
                                                   dropout=p_dropout, activation='gelu', 
                                                   batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, patch_size**2 * out_chans)
        
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.patch_embed(x) + self.pos_embed
        x = self.norm(self.blocks(x))
        x = self.decoder_pred(x)
        
        x = x.view(B, self.grid_h, self.grid_w, self.patch_size, self.patch_size, self.out_chans)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, self.out_chans, H, W)
        return x