import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class CavityDataset(Dataset):
    def __init__(self, data_dir='../../data/cavity', mode='train', model_type='fno'):
        """
        FNO Dataset Loader
        Input: (Batch, 50, 50, 3) -> [Kn, x, y]
        Output: (Batch, 50, 50, 10) -> [w, P, q]
        """
        self.model_type = model_type.lower()
        
        search_path = os.path.join(os.path.expanduser(data_dir), '*.npz')
        file_list = sorted(glob.glob(search_path))
        
        if len(file_list) == 0:
            raise FileNotFoundError(f"in {search_path} couldn't find .npz file！")
            
        print(f"[{model_type.upper()}] loading {len(file_list)} data files...")

        input_list = []
        target_list = []
        
        temp_data = np.load(file_list[0], allow_pickle=True)
        w_shape = temp_data['w'].shape 
        self.nx = w_shape[0] # 50
        self.ny = w_shape[1] # 50
        
        x = np.linspace(0, 1, self.nx)
        y = np.linspace(0, 1, self.ny)
        X, Y = np.meshgrid(x, y, indexing='ij') 
        self.grid_coords = np.stack([X, Y], axis=-1) # (50, 50, 2)
        
        for fname in file_list:
            data = np.load(fname, allow_pickle=True)
            
            base_name = os.path.basename(fname)
            try:
                kn_str = base_name.split('Kn')[1].replace('.npz', '')
                kn_value = float(kn_str)
            except:
                continue
            
            w = data['w']   # (50, 50, 4)
            q = data['q']   # (50, 50, 2)
            P = data['P']   # (50, 50, 2, 2)
            P_flat = P.reshape(self.nx, self.ny, -1)
            
            target_sample = np.concatenate([w, P_flat, q], axis=-1)

            kn_map = np.full((self.nx, self.ny, 1), kn_value, dtype=np.float32)
            
            input_image = np.concatenate([kn_map, self.grid_coords], axis=-1)
            
            input_list.append(input_image)
            target_list.append(target_sample)

        self.inputs = np.stack(input_list, axis=0)      # (99, 50, 50, 3)
        self.targets = np.stack(target_list, axis=0)    # (99, 50, 50, 10)
        
        self.target_mean = np.mean(self.targets, axis=(0,1,2))
        self.target_std = np.std(self.targets, axis=(0,1,2)) + 1e-6
        
        self.targets_norm = (self.targets - self.target_mean) / self.target_std
        
        kn_min, kn_max = 0.02, 1.0
        self.inputs[..., 0] = (self.inputs[..., 0] - kn_min) / (kn_max - kn_min)

        self.inputs_tensor = torch.tensor(self.inputs, dtype=torch.float32)
        self.targets_tensor = torch.tensor(self.targets_norm, dtype=torch.float32)
        
        print(f"done!: Input {self.inputs_tensor.shape}, Target {self.targets_tensor.shape}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs_tensor[idx], self.targets_tensor[idx]

if __name__ == "__main__":
    ds = CavityDataset()
    x, y = ds[0]
    print(f"Test Shape: Input {x.shape}, Target {y.shape}")
