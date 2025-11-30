import torch
import torch.nn as nn
import os
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import time

# --- Import your modules ---
from dataset import NYUDepthDataset 
from model import DepthModel

# --- Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
LR = 5e-5                   # LOWER Learning Rate for fine-tuning (was 1e-4)
SWAP_FREQUENCY = 5          
TOTAL_CHUNKS = 6
TOTAL_EPOCHS = TOTAL_CHUNKS * SWAP_FREQUENCY
CHECKPOINT_DIR = 'SecondPass/'

# --- CUSTOM LOSS FUNCTION (For Sharpness) ---
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
        
        # Combined Loss
        return loss_depth + (loss_dx + loss_dy)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Curriculum Depth Model")
    parser.add_argument('--start_chunk', type=int, default=0, 
                        help='Chunk index to start training from (0-5).')
    parser.add_argument('--resume_from', type=str, default=None, 
                        help='Path to a specific checkpoint to load weights from (e.g., checkpoints/checkpoint_chunk_5.pth)')
    return parser.parse_args()

def save_checkpoint(model, optimizer, epoch, chunk_idx, loss, filename):
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        
    checkpoint = {
        'epoch': epoch,
        'chunk_idx': chunk_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, filename))
    print(f"-> Checkpoint saved: {filename}")

def plot_loss_history(train_history, val_history, chunk_idx):
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        
    plt.figure(figsize=(10, 5))
    plt.plot(train_history, label='Train Loss')
    plt.plot(val_history, label='Val Loss')
    plt.title(f'Loss History (Up to End of Chunk {chunk_idx})')
    plt.xlabel('Epochs (Session)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(CHECKPOINT_DIR, f'loss_plot_chunk_{chunk_idx}.png')
    plt.savefig(save_path)
    plt.close()

def get_dataloader_for_chunk(chunk_idx):
    csv_path = f'data/splits/train_chunk_{chunk_idx}.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Chunk file not found: {csv_path}")

    print(f"\n[Curriculum] Loading Data Chunk: {chunk_idx}")
    # Root dir is '.' because we fixed the path double-stacking issue
    dataset = NYUDepthDataset(csv_file=csv_path, root_dir='.') 
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    return loader

def train():
    args = parse_args()
    start_chunk = args.start_chunk
    start_epoch = start_chunk * SWAP_FREQUENCY
    
    print(f"--- Starting Training Configuration ---")
    print(f"Start Chunk: {start_chunk}")
    print(f"Resume File: {args.resume_from if args.resume_from else 'None (Fresh Start)'}")
    print(f"Loss Function: EdgeLoss (L1 + Gradients)")
    print(f"Learning Rate: {LR} (Fine-tuning)")
    print(f"---------------------------------------")

    model = DepthModel().to(DEVICE)
    # Using EdgeLoss now to sharpen results
    criterion = EdgeLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler()
    
    # --- LOGIC TO LOAD WEIGHTS ---
    weights_loaded = False
    
    # 1. Explicit Resume (User provided a path) - HIGHEST PRIORITY
    if args.resume_from:
        if os.path.exists(args.resume_from):
            print(f"Loading weights from: {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            weights_loaded = True
            # NOTE: We do NOT load the optimizer state here. 
            # We want a fresh optimizer with a lower LR for the second pass.
        else:
            print(f"Error: Checkpoint {args.resume_from} not found!")
            return

    # 2. Implicit Resume (User started mid-way, e.g. chunk 3)
    elif start_chunk > 0:
        prev_ckpt = f"checkpoints/checkpoint_chunk_{start_chunk-1}.pth"
        if os.path.exists(prev_ckpt):
            print(f"Loading state from previous chunk: {prev_ckpt}")
            checkpoint = torch.load(prev_ckpt, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            weights_loaded = True

    if not weights_loaded:
        print("Starting with FRESH weights (Unless you meant to resume?)")

    # Validation loader
    val_dataset = NYUDepthDataset('data/val_split.csv', root_dir='.')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    train_loss_history = []
    val_loss_history = []
    current_loader = None
    
    for epoch in range(start_epoch, TOTAL_EPOCHS):
        current_chunk_idx = epoch // SWAP_FREQUENCY
        
        if epoch % SWAP_FREQUENCY == 0 or current_loader is None:
            current_loader = get_dataloader_for_chunk(current_chunk_idx)
        
        model.train()
        total_loss = 0
        
        for batch_idx, (imgs, depths) in enumerate(current_loader):
            imgs, depths = imgs.to(DEVICE), depths.to(DEVICE)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                preds = model(imgs)
                loss = criterion(preds, depths)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(current_loader)
        train_loss_history.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, depths in val_loader:
                imgs, depths = imgs.to(DEVICE), depths.to(DEVICE)
                preds = model(imgs)
                val_loss += criterion(preds, depths).item()
        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{TOTAL_EPOCHS} [Chunk {current_chunk_idx}] | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if (epoch + 1) % SWAP_FREQUENCY == 0:
            print(f"\n>>> Chunk {current_chunk_idx} Completed.")
            save_checkpoint(model, optimizer, epoch, current_chunk_idx, avg_train_loss, f"checkpoint_chunk_{current_chunk_idx}_pass2.pth")
            plot_loss_history(train_loss_history, val_loss_history, current_chunk_idx)
            
            if current_chunk_idx < TOTAL_CHUNKS - 1:
                # Auto-continue for 2nd pass usually, but keeping prompt for safety
                pass # You can remove the input() prompt here if you want it to run fully automatically

if __name__ == "__main__":
    train()