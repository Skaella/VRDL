import os
import argparse
import torch
import torch.nn as nn
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader import get_dataloaders
from model import PromptIR

def train():
    # 1. Path & CLI Setup
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_DATA_ROOT = os.path.join(SCRIPT_DIR, "dataset")
    DEFAULT_CKPT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
    DEFAULT_RESULT_DIR = os.path.join(SCRIPT_DIR, "results")

    parser = argparse.ArgumentParser(description="PromptIR HW4 Training")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--ckpt_dir", type=str, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--result_dir", type=str, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--batch_size", type=int, default=1) # Small for laptop VRAM
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Initialize PromptIR
    print(f"Initializing PromptIR on {device}...")
    model = PromptIR(
        dim=48, 
        num_blocks=[4, 6, 6, 8], 
        decoder=True # Ensures PromptGenBlock is active
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.L1Loss()
    
    # 3. Load Data
    train_loader, val_loader, _ = get_dataloaders(args.data_root, batch_size=args.batch_size)
    print(f"Dataset Loaded: {len(train_loader.dataset)} training samples.")

    metrics_history = []
    
    best_val_psnr = 0.0
    # 4. Training Loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            degraded = batch['degraded'].to(device)
            clean = batch['clean'].to(device)

            optimizer.zero_grad()
            restored = model(degraded)
            loss = criterion(restored, clean)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

           # Track metrics for visuals.py
        avg_loss = epoch_loss / len(train_loader)
        
        model.eval()
        total_val_psnr = 0.0
        
        #ImageNet constants to tensors for un-normalizing the range
        mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std_t = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        
        with torch.no_grad():
            for val_batch in val_loader:
                val_degraded = val_batch['degraded'].to(device)
                val_clean = val_batch['clean'].to(device)
                
                val_restored = model(val_degraded)
                
                # Un-normalize both back to [0, 1] to calculate accurate mathematical PSNR
                val_restored_unnorm = torch.clamp((val_restored * std_t) + mean_t, 0.0, 1.0)
                val_clean_unnorm = torch.clamp((val_clean * std_t) + mean_t, 0.0, 1.0)
                
                # Calculate Mean Squared Error and PSNR
                mse = torch.mean((val_restored_unnorm - val_clean_unnorm) ** 2)
                psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
                total_val_psnr += psnr.item()
                
        avg_val_psnr = total_val_psnr / len(val_loader)
        print(f"Epoch {epoch+1} Summary - Train Loss: {avg_loss:.4f} | Val PSNR: {avg_val_psnr:.2f} dB")
        
        # Append real values to generates an accurate graph
        metrics_history.append({"epoch": epoch + 1, "loss": avg_loss, "val_psnr": avg_val_psnr})

        # Save the Best Model 
        if avg_val_psnr > best_val_psnr:
            best_val_psnr = avg_val_psnr
            best_ckpt_path = os.path.join(args.ckpt_dir, "promptir_best.pth")
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"🌟 New Best Model Saved! Highest Val PSNR: {best_val_psnr:.2f} dB")

        # Periodic Save
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"promptir_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            
            # Save the metrics history to JSON for visuals.py tracking
            with open(os.path.join(args.result_dir, "metrics.json"), "w") as f:
                json.dump(metrics_history, f)
            print(f"Saved Periodic Checkpoint: {ckpt_path}")

if __name__ == "__main__":
    train()

