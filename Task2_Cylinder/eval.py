import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from model_deeponet import BoltzmannDeepONet
from model_fno import FNO2d
from model_unet import FluidUNet
from model_vit import VisionTransformer
from model_ae import Autoencoder
from model_pt import PointTransformer
from data_loader import CylinderDataset

def get_args():
    parser = argparse.ArgumentParser(description="Evaluation & Plotting Script for Task II: Cylinder Flow")
    parser.add_argument('--model', type=str, required=True, 
                        choices=['deeponet', 'fno', 'unet', 'vit', 'ae', 'pt'], 
                        help='Choose the baseline model to evaluate')
    parser.add_argument('--data_path', type=str, default='./data/cylinder_full_2400.pt', help='Path to dataset')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model_{}.pth', help='Path to weights')
    return parser.parse_args()

def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Starting Evaluation | Model: {args.model.upper()} | Device: {device}")

    if args.model in ['fno', 'unet', 'vit', 'ae']:
        data_mode = 'grid'
    elif args.model == 'deeponet':
        data_mode = 'deeponet'
    else:
        data_mode = 'point'

    dataset = CylinderDataset(args.data_path, mode=data_mode)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_data = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    
    from torch.utils.data import DataLoader
    # Prevent OOM for Point Transformer
    batch_size = 1 if args.model != 'pt' else 1
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Initialize model
    if args.model == 'fno':
        model = FNO2d(modes1=12, modes2=12, width=32, in_channels=4, out_channels=4)
    elif args.model == 'unet':
        model = FluidUNet(in_channels=4, out_channels=4, base_dim=19)
    elif args.model == 'vit':
        model = VisionTransformer(img_size=(128, 192), patch_size=8, in_chans=4, out_chans=4, embed_dim=144, depth=4)
    elif args.model == 'ae':
        model = Autoencoder(in_channels=4, out_channels=4, base_width=36)
    elif args.model == 'deeponet':
        model = BoltzmannDeepONet(branch_dim=2, trunk_dim=2, hidden_dim=280, num_outputs=4, depth=5)
    elif args.model == 'pt':
        model = PointTransformer(in_dim=4, out_dim=4, embed_dim=144, depth=4)
    
    model = model.to(device)
    
    ckpt_path = args.checkpoint.format(args.model)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"[SUCCESS] Loaded weights from {ckpt_path}")

    total_mae = 0.0
    total_l2_error = 0.0
    criterion_mae = nn.L1Loss(reduction='sum')
    
    plot_gt, plot_pred, plot_err = None, None, None

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if data_mode == 'grid':
                x, y = batch[0].to(device), batch[1].to(device)
                pred = model(x)
            elif data_mode == 'deeponet':
                x_branch, x_trunk, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                x_branch = x_branch[:, :2] 
                # Extract coordinates from the first sample
                x_trunk = x_trunk[0]
                pred = model(x_branch, x_trunk)
            elif data_mode == 'point':
                x, y = batch[0].to(device), batch[1].to(device)
                # Chunk processing to prevent OOM
                if x.shape[1] > 10000:
                    chunk_size = 8192
                    pred_chunks = []
                    for i in range(0, x.shape[1], chunk_size):
                        x_chunk = x[:, i:i+chunk_size, :]
                        pred_chunk = model(x_chunk)
                        pred_chunks.append(pred_chunk)
                    pred = torch.cat(pred_chunks, dim=1)
                else:
                    pred = model(x)

            total_mae += criterion_mae(pred, y).item()
            l2_err = torch.norm(pred - y, p=2) / (torch.norm(y, p=2) + 1e-8)
            total_l2_error += l2_err.item()

            # Select common sample for visualization
            if i == len(test_loader) - 20 or (i == 0 and len(test_loader) < 20):
                GRID_H, GRID_W = 128, 192
                
                if data_mode == 'grid':
                    pred_img = pred[0].permute(1, 2, 0).cpu().numpy()
                    y_img = y[0].permute(1, 2, 0).cpu().numpy()
                else:
                    pred_img = pred[0].view(GRID_H, GRID_W, 4).cpu().numpy()
                    y_img = y[0].view(GRID_H, GRID_W, 4).cpu().numpy()

                # Denormalize velocity u (channel 1)
                try:
                    # Handle Subset wrapper
                    base_dataset = test_data.dataset if hasattr(test_data, 'dataset') else dataset
                    u_true = base_dataset.denormalize(torch.tensor(y_img[..., 1]), 'u').numpy()
                    u_pred = base_dataset.denormalize(torch.tensor(pred_img[..., 1]), 'u').numpy()
                except (AttributeError, KeyError):
                    u_true, u_pred = y_img[..., 1], pred_img[..., 1]

                # Apply physical mask (filter cylinder interior)
                is_solid = (np.abs(y_img[..., 0]) < 1e-5)
                u_true[is_solid] = np.nan
                u_pred[is_solid] = np.nan
                
                plot_gt = u_true
                plot_pred = u_pred
                plot_err = np.abs(u_true - u_pred)
                plot_err[is_solid] = np.nan

    num_elements = len(test_loader.dataset) * np.prod(y.shape[1:])
    final_mae = total_mae / num_elements
    final_rel_l2 = total_l2_error / len(test_loader.dataset)

    print("-" * 50)
    print(f"[RESULTS] Final Results for {args.model.upper()}:")
    print(f"Normalized MAE      : {final_mae:.5f}")
    print(f"Relative L2 Error   : {final_rel_l2:.5f}")
    print("-" * 50)

    # Visualization
    plt.figure(figsize=(18, 5.5))
    plt.subplots_adjust(top=0.85, bottom=0.15, wspace=0.25)
    
    current_cmap = plt.cm.jet
    current_cmap.set_bad(color='white') 
    extent =[-3.0, 5.5, -3.0, 3.0]
    vmin, vmax = np.nanmin(plot_gt), np.nanmax(plot_gt)

    titles =[
        "Ground Truth (Velocity u)", 
        f"{args.model.upper()} Prediction\n(Rel L2 Error: {final_rel_l2*100:.2f}%)", 
        "Absolute Error"
    ]
    data_list = [plot_gt, plot_pred, plot_err]

    for i, (data, title) in enumerate(zip(data_list, titles)):
        ax = plt.subplot(1, 3, i+1)
        plt.title(title, fontsize=16, fontweight='bold', pad=15)
        im = plt.imshow(data, origin='lower', cmap=current_cmap, extent=extent, 
                        vmin=vmin if i<2 else None, vmax=vmax if i<2 else None)
        # Draw cylinder outline
        ax.add_patch(plt.Circle((0, 0), 0.5, color='black', fill=False, linewidth=2.0))
        ax.axis('off')
        
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=12)

    save_fig_path = f"evaluation_cylinder_{args.model}.png"
    plt.savefig(save_fig_path, dpi=300, bbox_inches='tight', transparent=False)
    print(f"[SAVED] Visualization saved to {save_fig_path}")

if __name__ == "__main__":
    main()