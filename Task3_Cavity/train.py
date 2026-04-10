import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split

from model_deeponet import BoltzmannDeepONet
from model_fno import FNO
from model_unet import UNet
from model_vit import VisionTransformer
from model_ae import AutoEncoder
from model_pt import PointTransformer
from data_loader import CavityDataset

def get_args():
    parser = argparse.ArgumentParser(description="TransportBench - Task III: Cavity Flow")
    parser.add_argument('--model', type=str, required=True, 
                        choices=['deeponet', 'fno', 'unet', 'vit', 'ae', 'pt'], 
                        help='Choose the baseline model')
    parser.add_argument('--epochs', type=int, default=2500, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='./data/cavity', help='Directory containing .npz data')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save models')
    return parser.parse_args()

def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Starting Task III Training | Model: {args.model.upper()} | Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"best_model_{args.model}.pth")

    dataset = CavityDataset(data_dir=args.data_dir, mode='train', model_type='fno')
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

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
    print(f"🧠 Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.L1Loss() 

    best_test_loss = float('inf')
    history = {'train_loss':[], 'test_loss':[]}

    print("🔥 Training Started...")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_acc = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # Input adaptation
            if args.model in ['unet', 'vit']:
                x_in = x.permute(0, 3, 1, 2) 
                pred = model(x_in)
                if args.model == 'unet':
                    pred = pred.permute(0, 2, 3, 1) 
            elif args.model == 'deeponet':
                B = x.shape[0]
                x_branch = x[:, 0, 0, 0:1]
                x_trunk = x[0, :, :, 1:3].reshape(-1, 2)
                y = y.view(B, 2500, 10) 
                pred = model(x_branch, x_trunk)
            else:
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
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                
                if args.model in ['unet', 'vit']:
                    x_in = x.permute(0, 3, 1, 2)
                    pred = model(x_in)
                    if args.model == 'unet': pred = pred.permute(0, 2, 3, 1)
                elif args.model == 'deeponet':
                    B = x.shape[0]
                    x_branch = x[:, 0, 0, 0:1]
                    x_trunk = x[0, :, :, 1:3].reshape(-1, 2)
                    y = y.view(B, 2500, 10)
                    pred = model(x_branch, x_trunk)
                else:
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

        if epoch % 50 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch}/{args.epochs}] | Train: {avg_train_loss:.5f} | Test: {avg_test_loss:.5f} | LR: {optimizer.param_groups[0]['lr']:.2e} | Time: {elapsed:.1f}s{saved_flag}")

    print(f"🎉 Training Complete! Best Test Loss: {best_test_loss:.5f}. Model saved to {save_path}")

if __name__ == "__main__":
    main()