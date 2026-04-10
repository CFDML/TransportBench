import torch
from torch.utils.data import Dataset

class CylinderDataset(Dataset):
    def __init__(self, pt_path, mode='deeponet'):
        """
        Cylinder flow dataset loader
        """
        super().__init__()
        self.mode = mode.lower()
        
        print(f"Loading cylinder data from {pt_path}...")
        data_dict = torch.load(pt_path)
        
        self.input_params = data_dict["input_params"]  # [N, 2] -> Kn, Ma
        self.flow_label   = data_dict["flow_label"]    # [N, 128, 192, 4] -> Flow fields
        self.mask         = data_dict["mask"]          # [N, 1, 128, 192] -> Cylinder mask
        self.grid_coords  = data_dict["grid_coords"]   # [N, 128, 192, 2] -> Coordinates
        self.stats        = data_dict["flow_stats"]    # Statistics for denormalization
        
        self.n_samples = self.flow_label.shape[0]
        self.H, self.W = self.flow_label.shape[1], self.flow_label.shape[2]
        print(f"Data loaded. Samples: {self.n_samples}, Grid: {self.H}x{self.W}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        target = self.flow_label[idx].permute(2, 0, 1)  # (128, 192, 4) -> (4, 128, 192)
        
        if self.mode == 'deeponet':
            branch_in = self.input_params[idx]  # [2] (Kn, Ma)
            trunk_in = self.grid_coords[idx].reshape(-1, 2)  # [24576, 2]
            target_flat = self.flow_label[idx].reshape(-1, 4)  # [24576, 4]
            return branch_in, trunk_in, target_flat

        elif self.mode == 'grid':
            # Grid mode for FNO, U-Net, ViT, AutoEncoder
            kn_ma = self.input_params[idx].view(2, 1, 1).expand(-1, self.H, self.W)
            coords = self.grid_coords[idx].permute(2, 0, 1)
            x_input = torch.cat([kn_ma, coords], dim=0)  # [4, H, W]
            return x_input, target
        
        elif self.mode == 'point':
            # Point mode for Point Transformer
            kn_ma = self.input_params[idx].view(2, 1, 1).expand(-1, self.H, self.W)
            coords = self.grid_coords[idx].permute(2, 0, 1)
            x_input = torch.cat([kn_ma, coords], dim=0)
            x_input = x_input.permute(1, 2, 0).reshape(-1, 4)  # [H*W, 4]
            target_flat = target.permute(1, 2, 0).reshape(-1, 4)  # [H*W, 4]
            return x_input, target_flat

    def denormalize(self, tensor, var_name):
        """Denormalize from [-1, 1] to physical values"""
        stat = self.stats[var_name]
        _min, _max = stat['min'], stat['max']
        return (tensor + 1.0) / 2.0 * (_max - _min) + _min