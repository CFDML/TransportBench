import torch
import torch.nn as nn
import time
import os
import argparse
from data_utils import get_dataloader_and_stats

from model_ae import AutoEncoder2d
from model_deeponet import DeepONet2d
from model_fno import FNO2d
from model_pt import PointTransformer
from model_unet import FluidUNet
from model_vit import VisionTransformer

def get_args():
    parser = argparse.ArgumentParser(description="Universal Golden Protocol Training Script")
    parser.add_argument('--model', type=str, required=True, choices=['ae', 'deeponet', 'fno', 'pt', 'unet', 'vit'])
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--no_fourier', action='store_true', help='Disable Fourier Encoding')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    return parser.parse_args()

def build_model(model_name, use_fourier):
    if model_name == 'ae': return AutoEncoder2d(in_channels=5, out_channels=4, features=128, use_fourier=use_fourier)
    elif model_name == 'deeponet': return DeepONet2d(in_channels=5, out_channels=4, basis_size=256, use_fourier=use_fourier)
    elif model_name == 'fno': return FNO2d(modes1=8, modes2=64, width=64, in_channels=5, out_channels=4, use_fourier=use_fourier)
    elif model_name == 'pt': return PointTransformer(in_channels=5, out_channels=4, latent_dim=512, num_latents=1024, depth=10, use_fourier=use_fourier)
    elif model_name == 'unet': return FluidUNet(in_channels=5, out_channels=4, features=64, use_fourier=use_fourier)
    elif model_name == 'vit': return VisionTransformer(in_channels=5, out_channels=4, embed_dim=512, depth=10, use_fourier=use_fourier)

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_fourier = not args.no_fourier
    fourier_suffix = "_fourier" if use_fourier else "_nofourier"
    
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
    save_path = os.path.join(args.save_dir, f'best_model_{args.model}{fourier_suffix}.pth')
    log_file = os.path.join(args.save_dir, f'train_{args.model}{fourier_suffix}.log')

    def log(msg):
        print(msg)
        with open(log_file, 'a') as f: f.write(msg + '\n')

    log(f"=== Training {args.model.upper()} | Fourier: {use_fourier} | Device: {device} ===")
    log("Strategy: TIME-DILATED GOLDEN PROTOCOL (Peak LR @ 40%, Weights start @ Ep800)")

    train_loader, test_loader, x_norm, y_norm = get_dataloader_and_stats(args.data_path, args.batch_size, device)
    
    model = build_model(args.model, use_fourier).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Warmup: Peak learning rate at 40% of training
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, 
        steps_per_epoch=len(train_loader), pct_start=0.4, anneal_strategy='cos'
    )

    loss_fn = nn.L1Loss(reduction='none')
    best_test_loss = float('inf')
    best_test_epoch = -1

    for ep in range(args.epochs):
        model.train()
        train_loss_val = 0.0

        # Delayed curriculum learning: Unrestricted exploration for the first 800 epochs
        if ep < 800:
            w_wall, w_near, w_p = 1.0, 1.0, 1.0
        else:
            progress = min(1.0, (ep - 800) / 1000.0)
            w_wall = 1.0 + 4.0 * progress
            w_near = 1.0 + 1.0 * progress
            w_p = 1.0 + 1.0 * progress 

        for x, y in train_loader:
            x_enc = x_norm.encode(x)
            y_enc = y_norm.encode(y)

            optimizer.zero_grad()
            out_enc = model(x_enc)

            # L1 Loss weight matrix
            raw_loss = loss_fn(out_enc, y_enc)
            w = torch.ones_like(raw_loss)
            w[:, :, 2, :] = w_wall      # Wall
            w[:, :, 1, :] = w_near      # Near-wall
            w[:, :, 3, :] = w_near      # Near-wall
            w[:, 3, :, :] *= w_p        # Pressure channel augmentation

            loss = (raw_loss * w).mean()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss_val += loss.item()

        train_loss_val /= len(train_loader)

        # Evaluation
        model.eval()
        test_loss_val = 0.0
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_enc_test = x_norm.encode(x_test)
                y_enc_test = y_norm.encode(y_test)
                out_enc_test = model(x_enc_test)
                
                # Unweighted L1 loss for validation
                raw_test_loss = loss_fn(out_enc_test, y_enc_test)
                test_loss_val += raw_test_loss.mean().item()
                
        test_loss_val /= len(test_loader)

        if ep % 50 == 0 or ep == args.epochs - 1:
            log(f"Ep {ep:04d} | LR: {scheduler.get_last_lr()[0]:.1e} | Train: {train_loss_val:.5f} | Test(Pure): {test_loss_val:.5f} | W_wall: {w_wall:.1f}")

        # Save best model after epoch 1000 to prevent early overfitting
        if ep >= 1000 and test_loss_val < best_test_loss:
            best_test_loss = test_loss_val
            best_test_epoch = ep
            
            torch.save({
                'model_state': model.state_dict(),
                'config': {'use_fourier': use_fourier},
                'epoch': ep,
                'best_test_loss': best_test_loss
            }, save_path)
            
            if ep % 50 == 0:
                log(f"  >>> [SAVED] Epoch {ep:04d} | Golden Best Test Loss: {best_test_loss:.6f} <<<")

    log(f"Finished. Golden Best Test Loss: {best_test_loss:.6f} discovered at Epoch {best_test_epoch}")

if __name__ == "__main__":
    main()