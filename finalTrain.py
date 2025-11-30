import torch
import torch.nn as nn
import os
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
import numpy as np
import time  # <--- Added for timing

# --- Import your modules ---
from dataset import NYUDepthDataset 
from model import DepthModel

# --- Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
LR = 5e-5                   # Low LR for stability
SWAP_FREQUENCY = 5          
TOTAL_CHUNKS = 10           # Including new data chunks
TOTAL_EPOCHS = TOTAL_CHUNKS * SWAP_FREQUENCY
CHECKPOINT_DIR = 'FinalTrain/'

# --- 1. THE WINNING LOSS FUNCTION (Edge Loss) ---
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

# --- 2. DATA AUGMENTATION ---
color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

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
    print(f"-> Saved checkpoint: {filename}")

# --- NEW: Loss Plotting Function ---
def plot_loss_history(train_history, val_history, chunk_idx):
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        
    plt.figure(figsize=(10, 5))
    plt.plot(train_history, label='Train Loss', color='blue')
    plt.plot(val_history, label='Val Loss', color='orange')
    plt.title(f'Loss History (Up to End of Chunk {chunk_idx})')
    plt.xlabel('Epochs (Session)')
    plt.ylabel('Edge Loss Value')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(CHECKPOINT_DIR, f'loss_plot_chunk_{chunk_idx}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"-> Saved loss graph: {save_path}")

def get_dataloader_for_chunk(chunk_idx):
    csv_path = f'data/splits/train_chunk_{chunk_idx}.csv'
    if not os.path.exists(csv_path):
        print(f"[Warning] Chunk {chunk_idx} not found! Did you run create_extra_splits.py?")
        raise FileNotFoundError(f"Missing {csv_path}")

    dataset = NYUDepthDataset(csv_file=csv_path, root_dir='.') 
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_chunk', type=int, default=6, help="Start at chunk 6 for new data")
    parser.add_argument('--resume_from', type=str, required=True, help="Path to best Edge Loss checkpoint")
    args = parser.parse_args()
    
    start_epoch = args.start_chunk * SWAP_FREQUENCY
    
    model = DepthModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler()
    criterion = EdgeLoss().to(DEVICE)
    
    if os.path.exists(args.resume_from):
        print(f"Loading best weights from: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        print(f"Error: Checkpoint {args.resume_from} not found.")
        return

    val_loader = DataLoader(NYUDepthDataset('data/val_split.csv', root_dir='.'), batch_size=BATCH_SIZE, shuffle=False)

    print(f"--- Starting Final Training Phase (Chunks {args.start_chunk}-{TOTAL_CHUNKS-1}) ---")
    current_loader = None
    
    # Lists to store history for plotting
    train_loss_history = []
    val_loss_history = []
    
    for epoch in range(start_epoch, TOTAL_EPOCHS):
        # Start Timer
        epoch_start_time = time.time()
        
        chunk_idx = epoch // SWAP_FREQUENCY
        
        if epoch % SWAP_FREQUENCY == 0 or current_loader is None:
            print(f"\n>>> Loading Chunk {chunk_idx}...")
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
        with torch.no_grad():
            for imgs, depths in val_loader:
                imgs, depths = imgs.to(DEVICE), depths.to(DEVICE)
                preds = model(imgs)
                val_loss += criterion(preds, depths).item()
        
        # Calculate Stats
        avg_train = total_loss / len(current_loader)
        avg_val = val_loss / len(val_loader)
        
        train_loss_history.append(avg_train)
        val_loss_history.append(avg_val)
        
        # Stop Timer
        epoch_duration = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1}/{TOTAL_EPOCHS} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | Time: {epoch_duration:.1f}s")

        if (epoch + 1) % SWAP_FREQUENCY == 0:
            save_checkpoint(model, optimizer, epoch, chunk_idx, avg_train, f"checkpoint_final_chunk_{chunk_idx}.pth")
            # Save the graph
            plot_loss_history(train_loss_history, val_loss_history, chunk_idx)

if __name__ == "__main__":
    train()