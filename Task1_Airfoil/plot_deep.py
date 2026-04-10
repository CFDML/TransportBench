import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from torch.utils.data import DataLoader, random_split

# 导入你的模型和数据集
from model import BoltzmannDeepONet
from data_loader import AirfoilDataset

# 设置全局字体格式，使其符合学术论文规范 (类似Times New Roman)
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['axes.titlesize'] = 16

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = 'data/airfoil_unified_128x128.pt'  # 替换为你的真实数据路径
    
    # 1. 为了完美画图，我们同时加载 DeepONet模式 和 FNO模式 的数据集
    # FNO 模式仅仅是为了借用它方便的 2D 坐标系 (X, Y) 和翼型掩膜 (Mask)
    dataset_don = AirfoilDataset(data_path, mode='deeponet')
    dataset_fno = AirfoilDataset(data_path, mode='fno')
    
    # 固定相同的随机种子 (42)，确保抽出的3个翼型样本和之前 FNO 画的完全一样
    train_size = int(0.8 * len(dataset_don))
    test_size = len(dataset_don) - train_size
    
    _, test_data_don = random_split(dataset_don, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    _, test_data_fno = random_split(dataset_fno, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    
    loader_don = DataLoader(test_data_don, batch_size=3, shuffle=False)
    loader_fno = DataLoader(test_data_fno, batch_size=3, shuffle=False)
    
    batch_don = next(iter(loader_don))
    batch_fno = next(iter(loader_fno))

    # 提取 DeepONet 需要的输入 (Branch & Trunk)
    x_branch, x_trunk = batch_don[0].to(device), batch_don[1].to(device)

    # 2. 初始化 DeepONet 模型并加载已保存的权重
    model = BoltzmannDeepONet(branch_dim=674, trunk_dim=2, hidden_dim=128, num_outputs=4).to(device)
    model_path = './checkpoints/best_model_deeponet.pth' # 替换为你实际的 DeepONet PTH 路径
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. 进行推理计算 (按照 train.py 中的逻辑，对每个样本分别前向传播避免显存爆炸)
    pred_list = []
    with torch.no_grad():
        for i in range(3):
            # x_branch[i:i+1]: [1, 674], x_trunk[i]: [16384, 2]
            pred_i = model(x_branch[i:i+1], x_trunk[i])  # 预测输出形状: [1, 16384, 4]
            pred_list.append(pred_i)
    pred_batch = torch.cat(pred_list, dim=0) # 拼接为 [3, 16384, 4]
    
    # 核心步骤：将坐标点的预测结果 Reshape 为 128x128 图像网格形式
    # 形状转换: [3, 16384, 4] -> [3, 128, 128, 4] -> 维度换位: [3, 4, 128, 128]
    pred_np = pred_batch.cpu().numpy().reshape(3, 128, 128, 4).transpose(0, 3, 1, 2)
    
    # 从 FNO dataset 中提取完美的空间网格 (X,Y)、真实值 (GT) 和 留白掩膜 (Mask)
    x_fno_np = batch_fno[0].cpu().numpy()  # [3, 3(mask, x, y), 128, 128]
    gt_np = batch_fno[1].cpu().numpy()     # [3, 4(rho, u, v, T), 128, 128]
    
    # 4. 创建 3x3 的绘图画布
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    
    # u 速度在通道索引为 1 (rho=0, u=1, v=2, T=3)
    channel_idx = 1  
    
    # 循环遍历 3 个样本
    for i in range(3):
        # 提取掩膜、X坐标、Y坐标
        mask = x_fno_np[i, 0, :, :]
        X = x_fno_np[i, 1, :, :]
        Y = x_fno_np[i, 2, :, :]
        
        # 提取真实值和 DeepONet 预测值
        gt_u = gt_np[i, channel_idx, :, :]
        pred_u = pred_np[i, channel_idx, :, :]
        
        # 应用掩膜：将翼型内部(mask==0)的数据设为 NaN，画图时会自动留白呈现完美空心
        gt_u[mask == 0] = np.nan
        pred_u[mask == 0] = np.nan
        
        # 计算绝对误差
        err_u = np.abs(gt_u - pred_u)
        
        # 计算该样本真实值的最大最小值，用于统一前两行的Colorbar
        vmin, vmax = np.nanmin(gt_u), np.nanmax(gt_u)
        err_max = np.nanmax(err_u)
        
        # --- 第 1 行：Ground Truth (DSMC) ---
        ax = axes[0, i]
        c1 = ax.pcolormesh(X, Y, gt_u, cmap='jet', vmin=vmin, vmax=vmax, shading='auto')
        ax.set_title(f"({chr(97+i)}) Geo {i+1}, DSMC", pad=15)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        fig.colorbar(c1, ax=ax, fraction=0.046, pad=0.04)

        # --- 第 2 行：Prediction (DeepONet) ---
        ax = axes[1, i]
        c2 = ax.pcolormesh(X, Y, pred_u, cmap='jet', vmin=vmin, vmax=vmax, shading='auto')
        ax.set_title(f"({chr(100+i)}) Geo {i+1}, Prediction (DeepONet)", pad=15)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        fig.colorbar(c2, ax=ax, fraction=0.046, pad=0.04)

        # --- 第 3 行：Absolute Error ---
        ax = axes[2, i]
        c3 = ax.pcolormesh(X, Y, err_u, cmap='jet', vmin=0, vmax=err_max, shading='auto')
        ax.set_title(f"({chr(103+i)}) Geo {i+1}, Absolute error", pad=15)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        fig.colorbar(c3, ax=ax, fraction=0.046, pad=0.04)

    # 保存高分辨率图片
    plt.savefig('fig_airfoil_deeponet.png', dpi=300, bbox_inches='tight')
    plt.savefig('fig_airfoil_deeponet.pdf', bbox_inches='tight')
    print("Plot saved as fig_airfoil_deeponet.png and .pdf")
    plt.show()

if __name__ == "__main__":
    main()