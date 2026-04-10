import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split


from model_deeponet import BoltzmannDeepONet
from model_fno import FNO2d
from model_unet import FluidUNet
from model_vit import VisionTransformer
from model_ae import AutoEncoder
from model_pt import PointTransformerONet
# Import custom Dataset
from data_loader import AirfoilDataset

def get_args():
    parser = argparse.ArgumentParser(description="TransportBench - Task I: Airfoil Flow")
    parser.add_argument('--model', type=str, required=True, 
                        choices=['deeponet', 'fno', 'unet', 'vit', 'ae', 'pt'], 
                        help='Choose the baseline model')
    parser.add_argument('--epochs', type=int, default=2500, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--data_path', type=str, default='data/airfoil_unified_128x128.pt', help='Path to dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save models')
    return parser.parse_args()

def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Starting Task I Training | Model: {args.model.upper()} | Device: {device}")

    # 1. Automatically create directory for saving model weights
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"best_model_{args.model}.pth")

    # 2. Determine Dataset mode based on model architecture (Grid-based vs Coordinate-based)
    # FNO/UNet/ViT/AE require image-like formats [B, 3, H, W]
    # DeepONet/PT require coordinate formats [B, 674] and [N_grid, 2]
    data_mode = 'fno' if args.model in['fno', 'unet', 'vit', 'ae'] else 'deeponet'
    
    dataset = AirfoilDataset(args.data_path, mode=data_mode)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # 3. Initialize model
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
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    # 4. Define optimizer and loss function (Masked L1 or MSE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.L1Loss() # Alternatively, use MSELoss based on reference papers

    # ================= Core: Logic for saving the best model =================
    best_test_loss = float('inf')
    history = {'train_loss':[], 'test_loss':[]}

    print("Training Started...")
    for epoch in range(args.epochs):
        # --- Train Phase ---
        model.train()
        train_loss_acc = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            
            if data_mode == 'fno':
                x, y = batch[0].to(device), batch[1].to(device)
                pred = model(x)
                mask = x[:, 0:1, :, :] # Extract geometry mask
                loss = criterion(pred * mask, y * mask)
            else:
                # Coordinate-based models (DeepONet, PT)
                x_branch, x_trunk, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                # x_branch: [Batch, 674]
                # x_trunk: [Batch, 16384, 2]
                # y: [Batch, 16384, 4]
                
                batch_size = x_branch.shape[0]
                n_points = x_trunk.shape[1]
                
                # Process each sample individually (handling DeepONet trunk batching limitations)
                pred_list = []
                for i in range(batch_size):
                    pred_i = model(x_branch[i:i+1], x_trunk[i])  # [1, N_points, 4]
                    pred_list.append(pred_i)
                pred = torch.cat(pred_list, dim=0)  # [Batch, N_points, 4]
                
                # Compute loss
                loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss_acc += loss.item()
            
        scheduler.step()
        avg_train_loss = train_loss_acc / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # --- Test Phase (Eval to save best model) ---
        model.eval()
        test_loss_acc = 0.0
        with torch.no_grad():
            for batch in test_loader:
                if data_mode == 'fno':
                    x, y = batch[0].to(device), batch[1].to(device)
                    pred = model(x)
                    mask = x[:, 0:1, :, :]
                    loss = criterion(pred * mask, y * mask)
                else:
                    x_branch, x_trunk, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    batch_size = x_branch.shape[0]
                    
                    # Process each sample individually
                    pred_list = []
                    for i in range(batch_size):
                        pred_i = model(x_branch[i:i+1], x_trunk[i])
                        pred_list.append(pred_i)
                    pred = torch.cat(pred_list, dim=0)
                    
                    loss = criterion(pred, y)
                    
                test_loss_acc += loss.item()
                
        avg_test_loss = test_loss_acc / len(test_loader)
        history['test_loss'].append(avg_test_loss)

        # 🌟 Core: If the current test loss is the historical minimum, overwrite and save the weights!
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), save_path)
            saved_flag = " [BEST SAVED]"
        else:
            saved_flag = ""

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {avg_train_loss:.5f} | Test Loss: {avg_test_loss:.5f} | LR: {optimizer.param_groups[0]['lr']:.2e}{saved_flag}")

    print(f"Training Complete! Best Test Loss: {best_test_loss:.5f}. Model saved to {save_path}")
    
    # Save loss history for visualization plotting
    np.save(os.path.join(args.save_dir, f"history_{args.model}.npy"), history)

if __name__ == "__main__":
    main()