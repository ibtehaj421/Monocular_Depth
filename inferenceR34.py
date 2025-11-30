import torch
import cv2
import numpy as np
import pandas as pd
import os
import argparse
from torchvision import transforms
from model import DepthModel

# --- Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Approximate Kinect Intrinsics (Scaled to 320x320)
K_MATRIX = np.array([
    [518.85 * 0.5, 0,            325.58 * 0.5],
    [0,            518.85 * 0.66, 253.74 * 0.66],
    [0,            0,            1.0]
])

def load_model(checkpoint_path):
    print(f"Loading ResNet-34 model from {checkpoint_path}...")
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

# --- INTERACTIVE VISUALIZER ---
class InteractiveState:
    def __init__(self, rgb_img, pred_map, gt_map=None):
        self.original_rgb = rgb_img.copy()
        self.pred_map = pred_map
        self.gt_map = gt_map
        self.points = []
        self.mode = 'rgb' # Modes: rgb, pred, gt, error
        
        # Pre-compute visualizations
        # Prediction (Inferno Colormap)
        pred_norm = cv2.normalize(pred_map, None, 0, 255, cv2.NORM_MINMAX)
        self.pred_viz = cv2.applyColorMap(pred_norm.astype(np.uint8), cv2.COLORMAP_INFERNO)
        
        # Ground Truth (Inferno Colormap)
        if gt_map is not None:
            gt_norm = cv2.normalize(gt_map, None, 0, 255, cv2.NORM_MINMAX)
            self.gt_viz = cv2.applyColorMap(gt_norm.astype(np.uint8), cv2.COLORMAP_INFERNO)
            
            # Error Map (Magma Colormap for visibility)
            diff = np.abs(gt_map - pred_map)
            diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
            self.err_viz = cv2.applyColorMap(diff_norm.astype(np.uint8), cv2.COLORMAP_MAGMA)
        else:
            self.gt_viz = np.zeros_like(self.pred_viz)
            self.err_viz = np.zeros_like(self.pred_viz)

        self.current_img = self.original_rgb.copy()

    def toggle_mode(self):
        modes = ['rgb', 'pred', 'gt', 'error']
        current_idx = modes.index(self.mode)
        self.mode = modes[(current_idx + 1) % len(modes)]
        self.refresh_display()

    def refresh_display(self):
        if self.mode == 'rgb': base = self.original_rgb
        elif self.mode == 'pred': base = self.pred_viz
        elif self.mode == 'gt': base = self.gt_viz
        else: base = self.err_viz
        
        self.current_img = base.copy()
        
        # Overlay Mode Text
        text = f"VIEW: {self.mode.upper()}"
        if self.mode == 'gt' and self.gt_map is None: text += " (N/A)"
        cv2.putText(self.current_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Redraw points
        for i, p in enumerate(self.points):
            cv2.circle(self.current_img, p, 5, (0, 255, 0), -1)
            if i > 0:
                cv2.line(self.current_img, self.points[i-1], p, (0, 255, 255), 2)
        
        # Redraw measurement if active
        if len(self.points) == 2:
            self.calc_and_draw(self.points[0], self.points[1])

        cv2.imshow("ResNet-34 Analysis", self.current_img)

    def calc_and_draw(self, p1, p2):
        # 1. Prediction Calculations
        h, w = self.pred_map.shape
        # Clamp to bounds
        p1 = (min(p1[0], w-1), min(p1[1], h-1))
        p2 = (min(p2[0], w-1), min(p2[1], h-1))
        
        d1_p = self.pred_map[p1[1], p1[0]]
        d2_p = self.pred_map[p2[1], p2[0]]
        
        P1_3d = pixel_to_3d(p1[0], p1[1], d1_p, K_MATRIX)
        P2_3d = pixel_to_3d(p2[0], p2[1], d2_p, K_MATRIX)
        dist_pred = np.linalg.norm(P1_3d - P2_3d)
        
        # 2. Ground Truth Calculations
        dist_gt = -1.0
        if self.gt_map is not None:
            d1_g = self.gt_map[p1[1], p1[0]]
            d2_g = self.gt_map[p2[1], p2[0]]
            
            P1_gt = pixel_to_3d(p1[0], p1[1], d1_g, K_MATRIX)
            P2_gt = pixel_to_3d(p2[0], p2[1], d2_g, K_MATRIX)
            dist_gt = np.linalg.norm(P1_gt - P2_gt)
            
            error_m = abs(dist_pred - dist_gt)
            error_pct = (error_m / (dist_gt + 1e-6)) * 100

        # 3. Draw on Screen
        label = f"Pred: {dist_pred:.2f}m"
        mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
        cv2.putText(self.current_img, label, (mid[0]-40, mid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 4. Terminal Output
        print(f"\n--- MEASUREMENT ({self.mode.upper()} View) ---")
        print(f"[PRED] Dist: {dist_pred:.4f}m | Depth: {d1_p:.2f}m -> {d2_p:.2f}m")
        if dist_gt != -1.0:
            print(f"[REAL] Dist: {dist_gt:.4f}m | Depth: {d1_g:.2f}m -> {d2_g:.2f}m")
            print(f"[DIFF] Error: {error_m:.4f}m ({error_pct:.1f}%)")
            
            # Show GT on screen if available
            label_gt = f"Real: {dist_gt:.2f}m"
            cv2.putText(self.current_img, label_gt, (mid[0]-40, mid[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) >= 2: self.points = []
            self.points.append((x, y))
            self.refresh_display()

def run_inference(model_path, image_path=None):
    model = load_model(model_path)
    gt_map = None
    
    # 1. Select Image
    if image_path is None:
        # Use Test Set
        if os.path.exists('data/nyu2_test.csv'):
            df = pd.read_csv('data/nyu2_test.csv', header=None)
        else:
            df = pd.read_csv('data/val_split.csv', header=None)
            
        row = df.sample(1).iloc[0]
        image_path = "data/nyu2_test/00059_colors.png"
        gt_path = "data/nyu2_test/00059_depth.png"
        
        # Load Ground Truth
        if os.path.exists(gt_path):
            gt_raw = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            # Smart Scaling
            if gt_raw.max() > 255:
                gt_map = gt_raw.astype('float32') / 1000.0
            else:
                gt_map = gt_raw.astype('float32') / 255.0 * 10.0
            gt_map = cv2.resize(gt_map, (320, 320), interpolation=cv2.INTER_NEAREST)

    print(f"Running inference on: {image_path}")
    
    # 2. Preprocess
    img_bgr = cv2.imread(image_path)
    if img_bgr is None: raise FileNotFoundError(f"Image not found: {image_path}")
    
    img_bgr_320 = cv2.resize(img_bgr, (320, 320))
    img_rgb = cv2.cvtColor(img_bgr_320, cv2.COLOR_BGR2RGB)
    img_tensor = transforms.ToTensor()(img_rgb)
    img_tensor = NORMALIZE(img_tensor).unsqueeze(0).to(DEVICE)
    
    # 3. Predict
    with torch.no_grad():
        pred_map = model(img_tensor).squeeze().cpu().numpy()

    # 4. Interactive Mode
    print("\n>>> RESNET-34 ANALYSIS TOOL <<<")
    print(" [Click] : Measure Distance")
    print(" [T Key] : Cycle Views (RGB -> PRED -> GT -> ERROR)")
    print(" [Q Key] : Quit")
    
    state = InteractiveState(img_bgr_320, pred_map, gt_map)
    cv2.imshow("ResNet-34 Analysis", state.current_img)
    cv2.setMouseCallback("ResNet-34 Analysis", state.mouse_callback)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('t'): state.toggle_mode()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default to Chunk 3 or 4 since that is where you are currently at
    parser.add_argument('--model', type=str, default='checkpoints_r34_finetune/finetuned_chunk_5.pth')
    parser.add_argument('--image', type=str, default=None)
    args = parser.parse_args()
    
    # Auto-find best checkpoint
    if not os.path.exists(args.model):
        for i in range(9, -1, -1):
            path = f'checkpoints_r34/checkpoint_r34_chunk_{i}.pth'
            if os.path.exists(path):
                print(f"Found latest checkpoint: {path}")
                args.model = path
                break
                
    run_inference(args.model, args.image)