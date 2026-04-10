"""
Data loader for Task 4: Double Cone Flow
Compatible with original data_utils.py logic
"""
import torch
from torch.utils.data import TensorDataset, DataLoader

class GaussianNormalizer:
    """Gaussian normalization for inputs and outputs"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.eps = 1e-6
    
    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)
    
    def decode(self, x):
        return x * (self.std + self.eps) + self.mean
    
    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self
    
    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}
    
    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

def get_dataloader_and_stats(data_path, batch_size, device):
    """
    Load dataset and compute normalization statistics
    Returns: train_loader, test_loader, x_norm, y_norm
    
    This function replicates the exact logic from original data_utils.py
    """
    print(f"Loading dataset from {data_path}...")
    data = torch.load(data_path, weights_only=False)
    x_data = data['x'].float()
    y_data = data['y'].float()
    
    print("Applying Standard Log10-transform to Pressure channel (Index 3)...")
    y_data[:, 3, :, :] = torch.log10(y_data[:, 3, :, :] + 1e-6)
    
    print("Computing Gaussian statistics...")
    x_mean = torch.mean(x_data, dim=(0, 2, 3), keepdim=True)
    x_std = torch.std(x_data, dim=(0, 2, 3), keepdim=True)
    y_mean = torch.mean(y_data, dim=(0, 2, 3), keepdim=True)
    y_std = torch.std(y_data, dim=(0, 2, 3), keepdim=True)
    
    n_total = x_data.shape[0]
    
    if n_total == 51:
        n_train = 45
        n_test = 6
        print(">>> Detected Full 51 Cases! Using 45 Train / 6 Test split. <<<")
    elif n_total == 32:
        n_train = 28
        n_test = 4
        print(">>> Warning: Only detected 32 Cases! Using 28 Train / 4 Test split. <<<")
    else:
        n_test = max(1, int(n_total * 0.1))
        n_train = n_total - n_test
        print(f">>> Detected {n_total} Cases. Using {n_train} Train / {n_test} Test. <<<")
    
    g_cpu = torch.Generator()
    g_cpu.manual_seed(42)
    indices = torch.randperm(n_total, generator=g_cpu)
    
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    print(f"Dataset Split -> Total: {n_total} | Train: {len(train_idx)} | Test: {len(test_idx)}")

    x_train, y_train = x_data[train_idx].to(device), y_data[train_idx].to(device)
    x_test, y_test = x_data[test_idx].to(device), y_data[test_idx].to(device)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    x_norm = GaussianNormalizer(x_mean, x_std).to(device)
    y_norm = GaussianNormalizer(y_mean, y_std).to(device)

    return train_loader, test_loader, x_norm, y_norm
