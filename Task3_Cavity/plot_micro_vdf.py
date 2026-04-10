import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
from torch.utils.data import DataLoader

# 设置全局字体格式，符合学术论文规范
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']

# 导入你的模型和数据集
from model_unet import UNet
from model_fno import FNO
from model_deeponet import BoltzmannDeepONet
from data_loader import CavityDataset

def get_args():
    parser = argparse.ArgumentParser(description="Microscopic VDF Reconstruction")
    parser.add_argument('--model', type=str, default='deeponet', help='Which model to use')
    parser.add_argument('--data_dir', type=str, default='./data/cavity', help='Data dir')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model_{}.pth', help='Checkpoint path')
    return parser.parse_args()

def parse_macro_vars(real_tensor):
    """从 10 通道的反归一化张量中提取宏观物理量"""
    rho = np.clip(real_tensor[..., 0], a_min=1e-3, a_max=None)
    u   = real_tensor[..., 1] / rho
    v   = real_tensor[..., 2] / rho
    E   = real_tensor[..., 3] / rho
    T   = np.clip(E - 0.5 * (u**2 + v**2), a_min=1e-3, a_max=None)
    
    Pxx = real_tensor[..., 4]
    Pxy = real_tensor[..., 5]
    Pyy = real_tensor[..., 7]
    qx  = real_tensor[..., 8]
    qy  = real_tensor[..., 9]
    
    return rho, u, v, T, Pxx, Pyy, Pxy, qx, qy

def reconstruct_vdf(rho, u, v, T, Pxx, Pyy, Pxy, qx, qy, v_range=4.0, grid_res=100):
    """利用 Grad 13-Moment 展开重构微观 2D 速度分布函数 f(vx, vy)"""
    R = 1.0 # 假设无量纲气体常数为 1.0
    p = rho * R * T
    
    vx = np.linspace(-v_range, v_range, grid_res)
    vy = np.linspace(-v_range, v_range, grid_res)
    Vx, Vy = np.meshgrid(vx, vy)
    
    cx = Vx - u
    cy = Vy - v
    c2 = cx**2 + cy**2
    
    f_eq = (rho / (2 * np.pi * R * T)) * np.exp(-c2 / (2 * R * T))
    stress_term = (Pxx * cx**2 + 2 * Pxy * cx * cy + Pyy * cy**2) / (2 * p * R * T)
    heat_term = (qx * cx + qy * cy) / (p * R * T) * (c2 / (2 * R * T) - 2.0)
    
    f_neq = f_eq * (1.0 + stress_term + heat_term)
    
    # 防止因极端截断误差出现负概率
    f_neq = np.clip(f_neq, a_min=0, a_max=None)
    return Vx, Vy, f_neq

def plot_3d_vdf(ax, Vx, Vy, f, title, z_max, cmap='magma'):
    """在指定的坐标系绘制 3D 表面与 2D 底部投影"""
    # 3D surface
    ax.plot_surface(Vx, Vy, f, cmap=cmap, linewidth=0, antialiased=True, alpha=0.9)
    
    # Bottom 2D projection
    ax.contourf(Vx, Vy, f, zdir='z', offset=0.0, cmap=cmap, alpha=0.5, levels=20)

    # 坐标与标题设定
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel(r"Velocity $v_x$", fontsize=12, labelpad=8)
    ax.set_ylabel(r"Velocity $v_y$", fontsize=12, labelpad=8)
    ax.set_zlabel(r"Probability $f$", fontsize=12, labelpad=10)
    
    # 强制 Z 轴从 0 到 z_max，以确保同一列的 GT 和 Pred 缩放一致
    ax.set_zlim(0, z_max)
    
    # 隐藏背景网格，提升学术感
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)

def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔬 Starting Microscopic Probing | Model: {args.model.upper()}")

    # 1. 加载数据 (我们统一使用 grid mode 以获取完整的 y_target)
    dataset = CavityDataset(data_dir=args.data_dir, mode='grid')
    x_input, y_target = dataset[0] # 取测试集的第一个样本
    x_input = x_input.unsqueeze(0).to(device)

    # 2. 加载模型
    if args.model == 'unet':
        model = UNet(n_channels=3, n_classes=10).to(device)
    elif args.model == 'fno':
        model = FNO(modes1=12, modes2=12, width=32, in_channels=3, out_channels=10).to(device)
    elif args.model == 'deeponet':
        # DeepONet需要特殊的输入格式
        # Branch: Kn值 [1]
        # Trunk: 所有网格点坐标 [2500, 2]
        kn_value = x_input[0, 0, 0, 0]  # 提取Kn值
        x_branch = kn_value.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1]
        
        # 提取所有网格点坐标
        coords = x_input[0, :, :, 1:3]  # [50, 50, 2]
        x_trunk = coords.reshape(-1, 2).to(device)  # [2500, 2]
        
        model = BoltzmannDeepONet(branch_dim=1, trunk_dim=2, hidden_dim=280, num_outputs=10, depth=5).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    ckpt_path = args.checkpoint.format(args.model)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # 3. 推理
    with torch.no_grad():
        if args.model == 'unet':
            # UNet需要 [B, C, H, W] 格式
            pred_norm = model(x_input.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        elif args.model == 'deeponet':
            pred_norm = model(x_branch, x_trunk).view(1, 10, 50, 50).permute(0, 2, 3, 1)
        elif args.model == 'vit':
            # VIT expects [B, C, H, W] and outputs [B, H, W, C]
            pred_norm = model(x_input.permute(0, 3, 1, 2))
        elif args.model == 'pt':
            # PT expects and outputs [B, H, W, C]
            pred_norm = model(x_input)
        elif args.model == 'ae':
            # AE handles permutation internally
            pred_norm = model(x_input)
        else:
            pred_norm = model(x_input)

    # 将格式统一为 [H, W, C] 即 [50, 50, 10]
    pred_norm = pred_norm.squeeze(0)
    y_target = y_target.to(device)

    # 4. 反归一化
    mean = torch.tensor(dataset.target_mean, device=device).view(1, 1, 10)
    std = torch.tensor(dataset.target_std, device=device).view(1, 1, 10)
    
    pred_real = (pred_norm * std + mean).cpu().numpy()
    gt_real = (y_target * std + mean).cpu().numpy()

    # 5. 选取探测点
    pt_center = (25, 25)  # A点：方腔中心 (平衡态)
    pt_corner = (45, 45)  # B点：右上角 (非平衡态)
    points = {"Center (Equilibrium)": pt_center, "Top-Right Corner (Non-Equil.)": pt_corner}

    # 6. 开始画 6 宫格 3D 图 (3 行 x 2 列)
    # 增加高度以容纳 3 行图片
    fig = plt.figure(figsize=(15, 20)) 
    plt.suptitle(f"Microscopic Kinetic VDF Reconstruction ({args.model.upper()})", 
                 fontsize=22, fontweight='bold', y=0.95)

    for col, (name, (px, py)) in enumerate(points.items()):
        
        # 提取当前点的宏观量 (Ground Truth)
        vars_gt = parse_macro_vars(gt_real[px:px+1, py:py+1, :])
        vars_gt = [v.item() for v in vars_gt]
        Vx, Vy, f_gt = reconstruct_vdf(*vars_gt)

        # 提取当前点的宏观量 (Prediction)
        vars_pr = parse_macro_vars(pred_real[px:px+1, py:py+1, :])
        vars_pr = [v.item() for v in vars_pr]
        _, _, f_pr = reconstruct_vdf(*vars_pr)

        # 计算绝对误差
        f_err = np.abs(f_gt - f_pr)

        # 确定 GT 和 Pred 的 Z 轴上限，以便两张图尺度完全对齐
        z_max_val = max(np.max(f_gt), np.max(f_pr)) * 1.05

        # --- 第一行: Ground Truth ---
        ax_gt = fig.add_subplot(3, 2, 1 + col, projection='3d')
        title_gt = f"{name}\nGround Truth" if col == 0 else f"{name}\nGround Truth"
        plot_3d_vdf(ax_gt, Vx, Vy, f_gt, title=title_gt, z_max=z_max_val, cmap='magma')

        # --- 第二行: Prediction ---
        ax_pr = fig.add_subplot(3, 2, 3 + col, projection='3d')
        plot_3d_vdf(ax_pr, Vx, Vy, f_pr, title="Prediction", z_max=z_max_val, cmap='magma')

        # --- 第三行: Absolute Error ---
        ax_err = fig.add_subplot(3, 2, 5 + col, projection='3d')
        # Error 误差图使用独立上限，色系改为 'plasma' 以作区分
        plot_3d_vdf(ax_err, Vx, Vy, f_err, title="Absolute Error", z_max=np.max(f_err)*1.05, cmap='plasma')

    # 调整边距，防止字三重叠
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05, wspace=0.15, hspace=0.25)
    
    save_path_png = f"fig_microscopic_vdf_6panel_{args.model}.png"
    save_path_pdf = f"fig_microscopic_vdf_6panel_{args.model}.pdf"
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_pdf, bbox_inches='tight')
    print(f"📸 惊艳的6宫格微观重构图已保存: {save_path_png}")

if __name__ == "__main__":
    main()