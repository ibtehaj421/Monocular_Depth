import torch
import cv2
import numpy as np
import pandas as pd
import os
import argparse
from torchvision import transforms
from model import DepthModel

# --- CONFIGURATION ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Approximate Kinect Intrinsics (Scaled to 320x320)
K_MATRIX = np.array([
    [518.85 * 0.5, 0,            325.58 * 0.5],
    [0,            518.85 * 0.66, 253.74 * 0.66],
    [0,            0,            1.0]
])

def load_model(checkpoint_path):
    print(f"Loading ResNet-34 from: {checkpoint_path}")
    model = DepthModel().to(DEVICE)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

def pixel_to_3d(u, v, depth_val, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) * depth_val / fx
    y = (v - cy) * depth_val / fy
    z = depth_val
    return np.array([x, y, z])

class UnifiedVisualizer:
    def __init__(self, rgb, pred, gt=None):
        self.h, self.w = 320, 320
        self.rgb = rgb
        self.pred_map = pred
        self.gt_map = gt
        self.points = []
        
        # 1. Prepare RGB
        # Ensure it's valid uint8
        if self.rgb.max() <= 1.0: self.rgb = (self.rgb * 255).astype(np.uint8)
        
        # 2. Prepare Prediction Heatmap
        pred_norm = cv2.normalize(pred, None, 0, 255, cv2.NORM_MINMAX)
        self.pred_viz = cv2.applyColorMap(pred_norm.astype(np.uint8), cv2.COLORMAP_INFERNO)
        
        # 3. Prepare GT Heatmap
        if gt is not None:
            gt_norm = cv2.normalize(gt, None, 0, 255, cv2.NORM_MINMAX)
            self.gt_viz = cv2.applyColorMap(gt_norm.astype(np.uint8), cv2.COLORMAP_INFERNO)
        else:
            # Black placeholder if no GT
            self.gt_viz = np.zeros_like(self.pred_viz)
            cv2.putText(self.gt_viz, "NO GT", (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # 4. Stitch them together
        # Layout: [ RGB | PRED | GT ]
        self.canvas_base = np.hstack([self.rgb, self.pred_viz, self.gt_viz])
        self.canvas_current = self.canvas_base.copy()
        
        # Add labels
        cv2.putText(self.canvas_base, "RGB Source", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(self.canvas_base, "Predicted Depth", (330, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(self.canvas_base, "Ground Truth", (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        self.refresh()

    def refresh(self):
        self.canvas_current = self.canvas_base.copy()
        
        # Draw points on ALL panels
        for idx, (x, y) in enumerate(self.points):
            self.draw_marker_everywhere(x, y, idx)

        # Draw line if 2 points exist
        if len(self.points) == 2:
            self.calc_and_draw_line()

        cv2.imshow("Combined Analysis Tool", self.canvas_current)

    def draw_marker_everywhere(self, x, y, idx):
        color = (0, 0, 255) if idx == 0 else (255, 0, 0) # Red for P1, Blue for P2
        
        # Draw on RGB (0 offset)
        cv2.circle(self.canvas_current, (x, y), 5, color, -1)
        
        # Draw on Pred (+320 offset)
        cv2.circle(self.canvas_current, (x + 320, y), 5, color, -1)
        
        # Draw on GT (+640 offset)
        cv2.circle(self.canvas_current, (x + 640, y), 5, color, -1)

    def calc_and_draw_line(self):
        p1 = self.points[0]
        p2 = self.points[1]
        
        # Clamp coords to be safe
        x1, y1 = min(p1[0], 319), min(p1[1], 319)
        x2, y2 = min(p2[0], 319), min(p2[1], 319)
        
        # 1. Get Depth from PREDICTION
        d1 = self.pred_map[y1, x1]
        d2 = self.pred_map[y2, x2]
        
        # Metric Math (Prediction)
        P1_3d = pixel_to_3d(x1, y1, d1, K_MATRIX)
        P2_3d = pixel_to_3d(x2, y2, d2, K_MATRIX)
        dist_pred = np.linalg.norm(P1_3d - P2_3d)
        
        # 2. Get Depth from GROUND TRUTH (if available)
        dist_gt_str = "N/A"
        dist_gt_val = -1.0
        
        if self.gt_map is not None:
            gt_d1 = self.gt_map[y1, x1]
            gt_d2 = self.gt_map[y2, x2]
            
            GT_P1 = pixel_to_3d(x1, y1, gt_d1, K_MATRIX)
            GT_P2 = pixel_to_3d(x2, y2, gt_d2, K_MATRIX)
            dist_gt_val = np.linalg.norm(GT_P1 - GT_P2)
            dist_gt_str = f"{dist_gt_val:.2f}m"

        # 3. Draw Lines on all panels
        offsets = [0, 320, 640]
        for off in offsets:
            pt1_draw = (p1[0] + off, p1[1])
            pt2_draw = (p2[0] + off, p2[1])
            cv2.line(self.canvas_current, pt1_draw, pt2_draw, (0, 255, 255), 2)
            
            # Label
            mid = ((pt1_draw[0] + pt2_draw[0])//2, (pt1_draw[1] + pt2_draw[1])//2)
            
            # Determine label text based on panel
            if off == 640: # GT Panel
                label = f"Real: {dist_gt_str}"
            else: # RGB and Pred Panel show Prediction
                label = f"Pred: {dist_pred:.2f}m"
                
            # Text background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(self.canvas_current, (mid[0], mid[1]-h), (mid[0]+w, mid[1]+5), (0,0,0), -1)
            cv2.putText(self.canvas_current, label, (mid[0], mid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        print(f"--- MEASUREMENT ---")
        print(f"Prediction Dist: {dist_pred:.4f}m")
        if dist_gt_val != -1.0:
            error = abs(dist_pred - dist_gt_val)
            pct = (error / (dist_gt_val + 1e-6)) * 100
            print(f"Real Dist:       {dist_gt_val:.4f}m")
            print(f"Error:           {error:.4f}m ({pct:.1f}%)")
        print(f"Depths (Pred):   {d1:.2f}m -> {d2:.2f}m")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Normalize X coordinate (Wrap to 0-320 range)
            # This handles clicks on ANY panel (RGB, Pred, or GT)
            real_x = x % 320
            real_y = y
            
            if len(self.points) >= 2:
                self.points = [] # Reset
            
            self.points.append((real_x, real_y))
            self.refresh()

def run(model_path):
    # Auto-select random image from Test Set
    if os.path.exists('data/nyu2_test.csv'):
        df = pd.read_csv('data/nyu2_test.csv', header=None)
    else:
        df = pd.read_csv('data/val_split.csv', header=None)
    
    # Pick a random row
    row = df.sample(1).iloc[0]
    rgb_path = row[0]
    gt_path = row[1]
    
    print(f"Processing: {rgb_path}")
    
    # Load RGB
    img_bgr = cv2.imread(rgb_path)
    img_bgr_320 = cv2.resize(img_bgr, (320, 320))
    img_rgb = cv2.cvtColor(img_bgr_320, cv2.COLOR_BGR2RGB)
    
    # Load GT
    gt_map = None
    if os.path.exists(gt_path):
        gt_raw = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if gt_raw.max() > 255: gt_map = gt_raw.astype('float32') / 1000.0
        else: gt_map = gt_raw.astype('float32') / 255.0 * 10.0
        gt_map = cv2.resize(gt_map, (320, 320), interpolation=cv2.INTER_NEAREST)

    # Predict
    model = load_model(model_path)
    tensor = transforms.ToTensor()(img_rgb)
    tensor = NORMALIZE(tensor).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        pred_map = model(tensor).squeeze().cpu().numpy()

    # Launch Visualizer
    print("\n>>> COMBINED TOOL <<<")
    print("Click on ANY panel (RGB, Pred, or GT) to select points.")
    print("The selection will sync across all three.")
    print("Press 'q' to quit.")
    
    viz = UnifiedVisualizer(img_bgr_320, pred_map, gt_map)
    cv2.setMouseCallback("Combined Analysis Tool", viz.mouse_callback)
    
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()
    
    # Find best model automatically
    ckpt = args.model
    if ckpt is None:
        # Check finetune folder first
        if os.path.exists('checkpoints_r34_finetune/finetuned_chunk_5.pth'):
            ckpt = 'checkpoints_r34_finetune/finetuned_chunk_5.pth'
        # Check base folder next
        elif os.path.exists('checkpoints_r34/checkpoint_r34_chunk_9.pth'):
            ckpt = 'checkpoints_r34/checkpoint_r34_chunk_9.pth'
            
    run(ckpt)