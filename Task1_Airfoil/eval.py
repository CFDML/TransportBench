import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

# Import models and data loader
from model import BoltzmannDeepONet
from model_fno import FNO2d
from model_unet import FluidUNet
from model_vit import VisionTransformer
from model_ae import AutoEncoder
from model_pt import PointTransformerONet
from data_loader import AirfoilDataset

def get_args():
    parser = argparse.ArgumentParser(description="Evaluation Script for Task I: Airfoil Flow")
    parser.add_argument('--model', type=str, required=True, 
                        choices=['deeponet', 'fno', 'unet', 'vit', 'ae', 'pt'], 
                        help='Choose the baseline model to evaluate')
    parser.add_argument('--data_path', type=str, default='data/airfoil_unified_128x128.pt', help='Path to dataset')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model_{}.pth', help='Path to weights')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='data/output', help='Directory to save visualizations')
    return parser.parse_args()

def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Starting Evaluation | Model: {args.model.upper()} | Device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load test dataset
    data_mode = 'fno' if args.model in ['fno', 'unet', 'vit', 'ae'] else 'deeponet'
    dataset = AirfoilDataset(args.data_path, mode=data_mode)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    # Must ensure the same manual_seed to maintain consistency between train and test splits
    _, test_data = random_split(dataset,[train_size, test_size], generator=torch.Generator().manual_seed(42))
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # 2. Initialize model
    if args.model == 'fno':
        model = FNO2d(modes1=12, modes2=12, width=28, in_channels=3, out_channels=4)
    elif args.model == 'unet':
        model = FluidUNet(in_channels=3, out_channels=4, base_dim=20)
    elif args.model == 'vit':
        model = VisionTransformer(embed_dim=144, depth=4)
    elif args.model == 'ae':
        model = AutoEncoder(in_channels=3, out_channels=4, base_dim=24)
    elif args.model == 'deeponet':
        model = BoltzmannDeepONet(branch_dim=674, trunk_dim=2, hidden_dim=128, num_outputs=4)
    elif args.model == 'pt':
        model = PointTransformerONet(hidden_dim=256, num_outputs=4)
    
    model = model.to(device)
    
    # 3. Load model weights
    ckpt_path = args.checkpoint.format(args.model)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Loaded weights from {ckpt_path}")

    # 4. Calculate evaluation metrics and collect samples for visualization
    total_mae = 0.0
    total_l2_error = 0.0
    criterion_mae = nn.L1Loss(reduction='sum')
    
    # Store multiple samples for visualization
    samples_to_plot = []
    var_names = ['rho', 'u', 'v', 'T']
    
    # Get the actual geometry masks from dataset for coordinate-based models
    if data_mode != 'fno':
        # Load the full dataset to access geometry masks
        full_dataset = AirfoilDataset(args.data_path, mode='fno')  # Use 'fno' mode to get masks
        test_indices = test_loader.dataset.indices  # Get test set indices

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if data_mode == 'fno':
                x, y = batch[0].to(device), batch[1].to(device)
                pred = model(x)
                mask = x[:, 0:1, :, :]
            else:
                x_branch, x_trunk, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                batch_size = x_branch.shape[0]
                
                # Process each sample individually for coordinate-based models
                pred_list = []
                for j in range(batch_size):
                    pred_j = model(x_branch[j:j+1], x_trunk[j])
                    pred_list.append(pred_j)
                pred = torch.cat(pred_list, dim=0)
                
                # Get the actual geometry mask for this sample
                actual_idx = test_indices[i]
                mask_2d = full_dataset.geo_mask[actual_idx].to(device)  # [1, 128, 128]
                # Reshape mask to match pred shape: [1, 16384, 1]
                mask = mask_2d.view(1, -1, 1)  # [1, 16384, 1]

            # Apply mask
            pred_masked = pred * mask
            y_masked = y * mask

            # Calculate Metrics
            total_mae += criterion_mae(pred_masked, y_masked).item()
            
            # Relative L2 Error
            l2_err = torch.norm(pred_masked - y_masked, p=2) / (torch.norm(y_masked, p=2) + 1e-8)
            total_l2_error += l2_err.item()

            # Collect samples for visualization
            if i < args.num_samples:
                if data_mode == 'fno':
                    # Shape: [1, 4, 128, 128]
                    gt_sample = y_masked[0].cpu().numpy()  # [4, 128, 128]
                    pred_sample = pred_masked[0].cpu().numpy()  # [4, 128, 128]
                    mask_sample = mask[0, 0].cpu().numpy()  # [128, 128]
                else:
                    # Shape: [1, 16384, 4] -> reshape to [4, 128, 128]
                    gt_sample = y_masked[0].view(128, 128, 4).permute(2, 0, 1).cpu().numpy()
                    pred_sample = pred_masked[0].view(128, 128, 4).permute(2, 0, 1).cpu().numpy()
                    # Get the actual 2D mask - mask_2d shape is [1, 128, 128]
                    mask_sample = mask_2d[0].cpu().numpy()  # [128, 128]
                
                samples_to_plot.append({
                    'gt': gt_sample,
                    'pred': pred_sample,
                    'mask': mask_sample,
                    'index': i
                })

    num_samples = len(test_loader.dataset)
    num_elements = num_samples * np.prod(y_masked.shape[1:]) # Total number of elements
    
    final_mae = total_mae / num_elements
    final_rel_l2 = total_l2_error / num_samples

    print("-" * 50)
    print(f"Final Results for {args.model.upper()}:")
    print(f"Mean Absolute Error (MAE) : {final_mae:.5f}")
    print(f"Relative L2 Error         : {final_rel_l2:.5f}")
    print("-" * 50)

    # 5. Generate visualizations for multiple samples and variables
    print(f"\nGenerating visualizations for {len(samples_to_plot)} samples...")
    
    for sample_data in samples_to_plot:
        sample_idx = sample_data['index']
        gt = sample_data['gt']  # [4, 128, 128]
        pred = sample_data['pred']  # [4, 128, 128]
        mask = sample_data['mask']  # [128, 128]
        
        # Create a figure for all 4 variables
        fig, axes = plt.subplots(4, 3, figsize=(15, 16))
        
        for var_idx, var_name in enumerate(var_names):
            gt_var = gt[var_idx]
            pred_var = pred[var_idx]
            err_var = np.abs(gt_var - pred_var)
            
            # Apply mask for visualization
            gt_var_masked = np.ma.masked_where(mask < 0.5, gt_var)
            pred_var_masked = np.ma.masked_where(mask < 0.5, pred_var)
            err_var_masked = np.ma.masked_where(mask < 0.5, err_var)
            
            # Determine common color scale for GT and Pred
            vmin = min(gt_var_masked.min(), pred_var_masked.min())
            vmax = max(gt_var_masked.max(), pred_var_masked.max())
            
            # Ground Truth
            im0 = axes[var_idx, 0].imshow(gt_var_masked, cmap='jet', vmin=vmin, vmax=vmax)
            axes[var_idx, 0].set_title(f"GT - {var_name}", fontsize=12)
            axes[var_idx, 0].axis('off')
            plt.colorbar(im0, ax=axes[var_idx, 0], fraction=0.046, pad=0.04)
            
            # Prediction
            im1 = axes[var_idx, 1].imshow(pred_var_masked, cmap='jet', vmin=vmin, vmax=vmax)
            axes[var_idx, 1].set_title(f"Pred - {var_name}", fontsize=12)
            axes[var_idx, 1].axis('off')
            plt.colorbar(im1, ax=axes[var_idx, 1], fraction=0.046, pad=0.04)
            
            # Error
            im2 = axes[var_idx, 2].imshow(err_var_masked, cmap='hot')
            axes[var_idx, 2].set_title(f"Error - {var_name}", fontsize=12)
            axes[var_idx, 2].axis('off')
            plt.colorbar(im2, ax=axes[var_idx, 2], fraction=0.046, pad=0.04)
        
        plt.suptitle(f"{args.model.upper()} - Sample {sample_idx}", fontsize=16, y=0.995)
        plt.tight_layout()
        
        save_fig_path = os.path.join(args.output_dir, f"{args.model}_sample_{sample_idx}.png")
        plt.savefig(save_fig_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_fig_path}")
    
    print(f"\nAll visualizations saved to {args.output_dir}/")

if __name__ == "__main__":
    main()