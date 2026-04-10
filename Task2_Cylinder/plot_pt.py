"""
Point Transformer Visualization Script for Cylinder Flow

Generates 3x3 grid visualization for Point Transformer model predictions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from torch.utils.data import DataLoader, Subset
import argparse

# 导入模型和数据集
from model_pt import PointTransformer
from data_loader import CylinderDataset

# 设置全局字体格式，符合学术论文规范
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['axes.titlesize'] = 16

def plot_pt(device, data_path, checkpoint_path, sample_indices, channel_idx=1, var_name='u'):
    """
    绘制 Point Transformer 模型的 3x3 九宫格图片
    """
    print(f"\n{'='*70}")
    print(f"Generating Point Transformer Visualization")
    print(f"{'='*70}\n")
    
    # 1. 加载两种模式的数据集
    # Point模式用于模型输入，Grid模式用于可视化
    dataset_pt = CylinderDataset(data_path, mode='point')
    dataset_grid = CylinderDataset(data_path, mode='grid')
    
    test_data_pt = Subset(dataset_pt, sample_indices)
    test_data_grid = Subset(dataset_grid, sample_indices)
    
    loader_pt = DataLoader(test_data_pt, batch_size=3, shuffle=False)
    loader_grid = DataLoader(test_data_grid, batch_size=3, shuffle=False)
    
    batch_pt = next(iter(loader_pt))
    batch_grid = next(iter(loader_grid))
    
    # Point Transformer 输入: [Batch, N_points, 4]
    x_pt, y_pt = batch_pt[0].to(device), batch_pt[1].to(device)
    
    # Grid 数据用于可视化
    x_grid = batch_grid[0].cpu().numpy()  # [3, 4, 128, 192]
    y_grid = batch_grid[1].cpu().numpy()  # [3, 4, 128, 192]

    # 2. 初始化 Point Transformer 模型 (使用CPU避免显存不足)
    print("⚠️  Using CPU for Point Transformer inference (GPU memory insufficient for 24576 points)")
    model = PointTransformer(in_dim=4, out_dim=4, embed_dim=144, depth=4).to('cpu')
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()

    # 3. 推理 (在CPU上逐样本处理)
    pred_list = []
    with torch.no_grad():
        for i in range(3):
            print(f"  Processing sample {i+1}/3...")
            # 处理单个样本: [24576, 4]
            pred_i = model(x_pt[i:i+1].cpu())  # [1, 24576, 4]
            pred_list.append(pred_i)
    pred_batch = torch.cat(pred_list, dim=0)  # [3, 24576, 4]
    
    # Reshape: [3, 24576, 4] -> [3, 128, 192, 4] -> [3, 4, 128, 192]
    pred_np = pred_batch.cpu().numpy().reshape(3, 128, 192, 4).transpose(0, 3, 1, 2)
    
    # 4. 提取坐标和参数
    kn_values = x_grid[:, 0, 0, 0]
    ma_values = x_grid[:, 1, 0, 0]
    X = x_grid[0, 2, :, :]
    Y = x_grid[0, 3, :, :]
    
    # 5. 创建 3x3 绘图画布
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    
    for i in range(3):
        gt_var = y_grid[i, channel_idx, :, :]
        pred_var = pred_np[i, channel_idx, :, :]
        
        err_var = np.abs(gt_var - pred_var)
        
        vmin, vmax = np.nanmin(gt_var), np.nanmax(gt_var)
        err_max = np.nanmax(err_var)
        
        param_str = f"Kn={kn_values[i]:.2f}, Ma={ma_values[i]:.2f}"
        
        # --- 第 1 行：Ground Truth ---
        ax = axes[0, i]
        c1 = ax.pcolormesh(X, Y, gt_var, cmap='jet', vmin=vmin, vmax=vmax, shading='auto')
        ax.set_title(f"({chr(97+i)}) {param_str}, DSMC", pad=15)
        ax.set_xlabel('X / m')
        ax.set_ylabel('Y / m')
        ax.set_aspect('equal')
        fig.colorbar(c1, ax=ax, fraction=0.046, pad=0.04)

        # --- 第 2 行：Prediction (Point Transformer) ---
        ax = axes[1, i]
        c2 = ax.pcolormesh(X, Y, pred_var, cmap='jet', vmin=vmin, vmax=vmax, shading='auto')
        ax.set_title(f"({chr(100+i)}) {param_str}, Prediction (PT)", pad=15)
        ax.set_xlabel('X / m')
        ax.set_ylabel('Y / m')
        ax.set_aspect('equal')
        fig.colorbar(c2, ax=ax, fraction=0.046, pad=0.04)

        # --- 第 3 行：Absolute Error ---
        ax = axes[2, i]
        c3 = ax.pcolormesh(X, Y, err_var, cmap='jet', vmin=0, vmax=err_max, shading='auto')
        ax.set_title(f"({chr(103+i)}) {param_str}, Absolute error", pad=15)
        ax.set_xlabel('X / m')
        ax.set_ylabel('Y / m')
        ax.set_aspect('equal')
        fig.colorbar(c3, ax=ax, fraction=0.046, pad=0.04)

    # 保存图片
    output_png = f'fig_cylinder_pt_{var_name}.png'
    output_pdf = f'fig_cylinder_pt_{var_name}.pdf'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"✓ Point Transformer plot saved: {output_png} and {output_pdf}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Cylinder Flow Visualization for Point Transformer")
    parser.add_argument('--data_path', type=str, default='data/cylinder_full_2400.pt',
                        help='Path to dataset')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model_pt.pth',
                        help='Path to Point Transformer checkpoint')
    parser.add_argument('--samples', type=int, nargs=3, default=[100, 1000, 2300],
                        help='Three sample indices to visualize')
    parser.add_argument('--channel', type=int, default=1,
                        help='Channel index to visualize (0:rho, 1:u, 2:v, 3:T)')
    parser.add_argument('--var_name', type=str, default='u',
                        help='Variable name for output filename')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    channel_names = {0: 'rho', 1: 'u', 2: 'v', 3: 'T'}
    var_name = args.var_name if args.var_name else channel_names.get(args.channel, 'var')
    
    plot_pt(device, args.data_path, args.checkpoint, 
            args.samples, args.channel, var_name)
    
    print(f"\n{'='*70}")
    print("Point Transformer visualization completed successfully!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
