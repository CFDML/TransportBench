import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from torch.utils.data import DataLoader, random_split

# 导入你的模型和数据集 (请确保路径和类名与你的代码一致)
from model_fno import FNO2d
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
    
    # 1. 加载测试数据集
    data_path = 'data/airfoil_unified_128x128.pt'  # 替换为你的真实数据路径
    dataset = AirfoilDataset(data_path, mode='fno')
    
    # 固定随机种子以确保每次画图抽取的样本一致
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_data = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    
    # 我们不需要打乱，直接按顺序取出前3个具有明显几何差异的样本
    test_loader = DataLoader(test_data, batch_size=3, shuffle=False)
    batch = next(iter(test_loader))
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)

    # 2. 初始化模型并加载已保存的权重
    model = FNO2d(modes1=12, modes2=12, width=28, in_channels=3, out_channels=4).to(device)
    model_path = './checkpoints/best_model_fno.pth' # 替换为你实际的PTH路径
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. 进行推理计算
    with torch.no_grad():
        pred_batch = model(x_batch)
    
    # 将数据转移到CPU并转为numpy数组
    x_np = x_batch.cpu().numpy()       # [3, 3(mask, x, y), 128, 128]
    gt_np = y_batch.cpu().numpy()      # [3, 4(rho, u, v, T), 128, 128]
    pred_np = pred_batch.cpu().numpy() # [3, 4, 128, 128]
    
    # 4. 创建 3x3 的绘图画布
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    
    # u 速度在通道索引为 1 (rho=0, u=1, v=2, T=3)
    channel_idx = 1  
    
    # 循环遍历 3 个样本 (对应 3 列)
    for i in range(3):
        # 提取掩膜、X坐标、Y坐标
        mask = x_np[i, 0, :, :]
        X = x_np[i, 1, :, :]
        Y = x_np[i, 2, :, :]
        
        # 提取真实值和预测值
        gt_u = gt_np[i, channel_idx, :, :]
        pred_u = pred_np[i, channel_idx, :, :]
        
        # 应用掩膜：将翼型内部(mask==0)的数据设为 NaN，画图时会自动留白
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

        # --- 第 2 行：Prediction (FNO) ---
        ax = axes[1, i]
        c2 = ax.pcolormesh(X, Y, pred_u, cmap='jet', vmin=vmin, vmax=vmax, shading='auto')
        ax.set_title(f"({chr(100+i)}) Geo {i+1}, Prediction", pad=15)
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
    plt.savefig('fig_airfoil_generalization.png', dpi=300, bbox_inches='tight')
    plt.savefig('fig_airfoil_generalization.pdf', bbox_inches='tight')
    print("Plot saved as fig_airfoil_generalization.png and .pdf")
    plt.show()

if __name__ == "__main__":
    main()