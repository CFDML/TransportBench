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
from model_ae import Autoencoder
from model_pt import PointTransformer
from data_loader import CylinderDataset

def get_args():
    parser = argparse.ArgumentParser(description="TransportBench - Task II: Cylinder Flow Training")
    parser.add_argument('--model', type=str, required=True, 
                        choices=['deeponet', 'fno', 'unet', 'vit', 'ae', 'pt'], 
                        help='Choose the baseline model')
    parser.add_argument('--epochs', type=int, default=2500, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--data_path', type=str, default='./data/cylinder_full_2400.pt', help='Path to dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save models')
    return parser.parse_args()

def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Starting Task II Training | Model: {args.model.upper()} | Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"best_model_{args.model}.pth")

    # Determine data loading mode based on model architecture
    if args.model in ['fno', 'unet', 'vit', 'ae']:
        data_mode = 'grid'
    elif args.model == 'deeponet':
        data_mode = 'deeponet'
    else:
        data_mode = 'point'
        
    dataset = CylinderDataset(args.data_path, mode=data_mode)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Initialize model with configurations matching checkpoints
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
    print(f"🧠 Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    best_test_loss = float('inf')
    history = {'train_loss':[], 'test_loss':[]}

    print("🔥 Training Started...")
    for epoch in range(args.epochs):
        model.train()
        train_loss_acc = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            
            if data_mode == 'grid':
                x, y = batch[0].to(device), batch[1].to(device)
                pred = model(x)
            elif data_mode == 'deeponet':
                x_branch, x_trunk, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                x_branch = x_branch[:, :2]  # Ensure input is (Kn, Ma)
                x_trunk = x_trunk[0]  # All samples share the same grid
                pred = model(x_branch, x_trunk)
            elif data_mode == 'point':
                x, y = batch[0].to(device), batch[1].to(device)
                pred = model(x)

            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss_acc += loss.item()
            
        scheduler.step()
        avg_train_loss = train_loss_acc / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        test_loss_acc = 0.0
        with torch.no_grad():
            for batch in test_loader:
                if data_mode == 'grid':
                    x, y = batch[0].to(device), batch[1].to(device)
                    pred = model(x)
                elif data_mode == 'deeponet':
                    x_branch, x_trunk, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    x_branch = x_branch[:, :2]
                    x_trunk = x_trunk[0]
                    pred = model(x_branch, x_trunk)
                elif data_mode == 'point':
                    x, y = batch[0].to(device), batch[1].to(device)
                    pred = model(x)
                    
                loss = criterion(pred, y)
                test_loss_acc += loss.item()
                
        avg_test_loss = test_loss_acc / len(test_loader)
        history['test_loss'].append(avg_test_loss)

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), save_path)
            saved_flag = " 💾[BEST SAVED]"
        else:
            saved_flag = ""

        print(f"Epoch[{epoch+1}/{args.epochs}] | Train: {avg_train_loss:.5f} | Test: {avg_test_loss:.5f} | LR: {optimizer.param_groups[0]['lr']:.2e}{saved_flag}")

    print(f"🎉 Training Complete! Best Test Loss: {best_test_loss:.5f}. Model saved to {save_path}")

if __name__ == "__main__":
    main()