import torch
import torch.nn as nn
import os
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
import numpy as np
import time

# --- Import your modules ---
from dataset import NYUDepthDataset 
from model import DepthModel

# --- CONFIGURATION ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
LR = 1e-5                   # <--- 10x LOWER LEARNING RATE (Crucial for fine-tuning)
SWAP_FREQUENCY = 5          
TOTAL_CHUNKS = 10           # Ensure this matches your total split count
TOTAL_EPOCHS = TOTAL_CHUNKS * SWAP_FREQUENCY
CHECKPOINT_DIR = 'checkpoints_r34_finetune/' # Separate folder to keep organized

# --- LOSS FUNCTION (Edge Loss) ---
class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def gradient_x(self, img):
        img = torch.nn.functional.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:] 
        return gx

    def gradient_y(self, img):
        img = torch.nn.functional.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gy

    def forward(self, pred, target):
        loss_depth = self.l1(pred, target)
        pred_dx = self.gradient_x(pred)
        pred_dy = self.gradient_y(pred)
        target_dx = self.gradient_x(target)
        target_dy = self.gradient_y(target)
        loss_dx = self.l1(pred_dx, target_dx)
        loss_dy = self.l1(pred_dy, target_dy)
        return loss_depth + (loss_dx + loss_dy)

# --- AUGMENTATION ---
# Slightly gentler augmentation for fine-tuning
color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)

def apply_augmentation(image, depth):
    if torch.rand(1) < 0.5:
        image = torch.flip(image, [3])
        depth = torch.flip(depth, [3])
    if torch.rand(1) < 0.5:
        image = color_jitter(image)
    return image, depth

def save_checkpoint(model, optimizer, epoch, chunk_idx, loss, filename):
    if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)
    torch.save({
        'epoch': epoch,
        'chunk_idx': chunk_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, os.path.join(CHECKPOINT_DIR, filename))
    print(f"-> Saved: {filename}")

def plot_loss(train_hist, val_hist, chunk_idx):
    if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)
    if len(train_hist) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(train_hist, label='Train Loss', color='blue')
        plt.plot(val_hist, label='Val Loss', color='orange')
        plt.title(f'Fine-Tuning History (Up to Chunk {chunk_idx})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(CHECKPOINT_DIR, f'loss_plot_chunk_{chunk_idx}.png'))
        plt.close()

def get_dataloader_for_chunk(chunk_idx):
    csv_path = f'data/splits/train_chunk_{chunk_idx}.csv'
    # Safety loop if chunks don't exist
    if not os.path.exists(csv_path):
        csv_path = f'data/splits/train_chunk_{chunk_idx % 6}.csv'
    dataset = NYUDepthDataset(csv_file=csv_path, root_dir='.') 
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

def train():
    parser = argparse.ArgumentParser()
    # Resume from your BEST checkpoint from the previous run
    parser.add_argument('--resume_from', type=str, required=True)
    args = parser.parse_args()
    
    print(f"--- Starting Fine-Tuning (LR={LR}) ---")
    print(f"Saving to: {CHECKPOINT_DIR}")
    
    model = DepthModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # --- ADAPTIVE LR SCHEDULER (Cosine Annealing) ---
    # Slowly lowers LR from 1e-5 to 1e-7 over the full duration
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-7)
    
    scaler = torch.amp.GradScaler()
    criterion = EdgeLoss().to(DEVICE)

    # Load Weights
    if os.path.exists(args.resume_from):
        print(f"Loading Phase 1 weights: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        # NOTE: We do NOT load the optimizer state. We want a fresh optimizer with low LR.
    else:
        print(f"Error: {args.resume_from} not found.")
        return

    val_loader = DataLoader(NYUDepthDataset('data/nyu2_test.csv', root_dir='.'), batch_size=BATCH_SIZE, shuffle=False)

    current_loader = None
    train_hist = []
    val_hist = []
    
    # We iterate through all chunks (0 to 9) again
    for epoch in range(0, TOTAL_EPOCHS):
        start_time = time.time()
        chunk_idx = epoch // SWAP_FREQUENCY
        
        if epoch % SWAP_FREQUENCY == 0 or current_loader is None:
            print(f"\n>>> Fine-Tuning on Chunk {chunk_idx}...")
            current_loader = get_dataloader_for_chunk(chunk_idx)
        
        model.train()
        total_loss = 0
        
        for imgs, depths in current_loader:
            imgs, depths = imgs.to(DEVICE), depths.to(DEVICE)
            imgs, depths = apply_augmentation(imgs, depths)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                preds = model(imgs)
                loss = criterion(preds, depths)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        val_mae = 0
        with torch.no_grad():
            for imgs, depths in val_loader:
                imgs, depths = imgs.to(DEVICE), depths.to(DEVICE)
                preds = model(imgs)
                # Track Optimization Loss
                val_loss += criterion(preds, depths).item()
                # Track Pure Distance Error (L1) for reporting
                val_mae += torch.nn.L1Loss()(preds, depths).item()
        
        avg_train = total_loss / len(current_loader)
        avg_val = val_loss / len(val_loader)
        avg_mae = val_mae / len(val_loader)
        
        train_hist.append(avg_train)
        val_hist.append(avg_val)
        
        # Step the scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        duration = time.time() - start_time
        print(f"FT Epoch {epoch+1}/{TOTAL_EPOCHS} | Train: {avg_train:.4f} | Val Total: {avg_val:.4f} | Val MAE: {avg_mae:.4f}m | LR: {current_lr:.2e} | Time: {duration:.1f}s")

        if (epoch + 1) % SWAP_FREQUENCY == 0:
            save_checkpoint(model, optimizer, epoch, chunk_idx, avg_train, f"finetuned_chunk_{chunk_idx}.pth")
            plot_loss(train_hist, val_hist, chunk_idx)

if __name__ == "__main__":
    train()