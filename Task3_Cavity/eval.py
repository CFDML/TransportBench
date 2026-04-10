import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from model_deeponet import BoltzmannDeepONet
from model_fno import FNO
from model_unet import UNet
from model_vit import VisionTransformer
from model_ae import AutoEncoder
from model_pt import PointTransformer
from data_loader import CavityDataset

def get_args():
    parser = argparse.ArgumentParser(description="Evaluation Script for Task III: Cavity Flow")
    parser.add_argument('--model', type=str, required=True, 
                        choices=['deeponet', 'fno', 'unet', 'vit', 'ae', 'pt'], help='Choose model')
    parser.add_argument('--data_dir', type=str, default='./data/cavity', help='Path to .npz data')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model_{}.pth', help='Path to weights')
    return parser.parse_args()

def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Prevent OOM for Point Transformer
    if args.model == 'pt': device = 'cpu'
        
    print(f"📊 Starting Evaluation | Model: {args.model.upper()} | Device: {device}")

    dataset = CavityDataset(data_dir=args.data_dir, mode='test', model_type='fno')
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_data = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    if args.model == 'fno':
        model = FNO(modes1=12, modes2=12, width=32, in_channels=3, out_channels=10)
    elif args.model == 'unet':
        model = UNet(n_channels=3, n_classes=10)
    elif args.model == 'vit':
        model = VisionTransformer(img_size=50, patch_size=5, in_chans=3, out_chans=10, embed_dim=144, depth=4)
    elif args.model == 'ae':
        model = AutoEncoder(in_channels=3, out_channels=10, base_dim=32)
    elif args.model == 'deeponet':
        model = BoltzmannDeepONet(branch_dim=1, trunk_dim=2, hidden_dim=280, num_outputs=10, depth=5)
    elif args.model == 'pt':
        model = PointTransformer(in_channels=3, out_channels=10, embed_dim=120, depth=4)
    
    model = model.to(device)
    
    ckpt_path = args.checkpoint.format(args.model)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    total_mae, total_l2_error = 0.0, 0.0
    criterion_mae = nn.L1Loss(reduction='sum')
    
    plot_gt, plot_pred = None, None

    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            
            if args.model in['unet', 'vit']:
                x_in = x.permute(0, 3, 1, 2)
                pred = model(x_in)
                if args.model == 'unet': pred = pred.permute(0, 2, 3, 1)
            elif args.model == 'deeponet':
                B = x.shape[0]
                x_branch = x[:, 0, 0, 0:1]
                x_trunk = x[0, :, :, 1:3].reshape(-1, 2)
                pred = model(x_branch, x_trunk).view(B, 50, 50, 10)
            else:
                pred = model(x)

            total_mae += criterion_mae(pred, y).item()
            l2_err = torch.norm(pred - y, p=2) / (torch.norm(y, p=2) + 1e-8)
            total_l2_error += l2_err.item()

            if i == 0:
                mean = torch.tensor(dataset.target_mean, device=device).view(1, 1, 1, 10)
                std = torch.tensor(dataset.target_std, device=device).view(1, 1, 1, 10)
                
                pred_real = pred * std + mean
                target_real = y * std + mean
                
                plot_pred = pred_real[0].cpu().numpy()
                plot_gt = target_real[0].cpu().numpy()

    num_elements = len(test_loader.dataset) * np.prod(y.shape[1:])
    final_mae = total_mae / num_elements
    final_rel_l2 = total_l2_error / len(test_loader.dataset)

    print("-" * 50)
    print(f"🏆 Results for {args.model.upper()}: MAE = {final_mae:.5f} | Rel L2 = {final_rel_l2*100:.2f}%")

    nx, ny = dataset.nx, dataset.ny
    vx = np.linspace(0, 1, nx)
    vy = np.linspace(0, 1, ny)
    
    # Extract u, v and calculate velocity magnitude
    rho_t, u_t, v_t = plot_gt[..., 0], plot_gt[..., 1]/(plot_gt[..., 0]+1e-8), plot_gt[..., 2]/(plot_gt[..., 0]+1e-8)
    rho_p, u_p, v_p = plot_pred[..., 0], plot_pred[..., 1]/(plot_pred[..., 0]+1e-8), plot_pred[..., 2]/(plot_pred[..., 0]+1e-8)
    mag_t = np.sqrt(u_t**2 + v_t**2)
    mag_p = np.sqrt(u_p**2 + v_p**2)

    fig = plt.figure(figsize=(12, 10))
    
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title("Ground Truth (Full)", fontsize=14, fontweight='bold')
    ax1.imshow(mag_t.T, origin='lower', extent=[0,1,0,1], cmap='jet', vmin=0, vmax=0.15)
    ax1.streamplot(vx, vy, u_t.T, v_t.T, color='white', density=1.5, linewidth=0.8, arrowsize=0.8)
    
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title(f"{args.model.upper()} Prediction (Full)", fontsize=14, fontweight='bold')
    ax2.imshow(mag_p.T, origin='lower', extent=[0,1,0,1], cmap='jet', vmin=0, vmax=0.15)
    ax2.streamplot(vx, vy, u_p.T, v_p.T, color='white', density=1.5, linewidth=0.8, arrowsize=0.8)
    
    zoom_lim =[0.0, 0.4]
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_title("Ground Truth (Corner Zoom)", fontsize=14, fontweight='bold')
    ax3.imshow(mag_t.T, origin='lower', extent=[0,1,0,1], cmap='jet', vmin=0, vmax=0.02)
    ax3.streamplot(vx, vy, u_t.T, v_t.T, color='white', density=3.0, linewidth=1.0)
    ax3.set_xlim(zoom_lim); ax3.set_ylim(zoom_lim)
    
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_title(f"{args.model.upper()} Prediction (Corner Zoom)", fontsize=14, fontweight='bold')
    ax4.imshow(mag_p.T, origin='lower', extent=[0,1,0,1], cmap='jet', vmin=0, vmax=0.02)
    ax4.streamplot(vx, vy, u_p.T, v_p.T, color='white', density=3.0, linewidth=1.0)
    ax4.set_xlim(zoom_lim); ax4.set_ylim(zoom_lim)
    
    plt.tight_layout()
    save_fig_path = f"evaluation_cavity_{args.model}.png"
    plt.savefig(save_fig_path, dpi=300)
    print(f"📸 plot saved as: {save_fig_path}")

if __name__ == "__main__":
    main()