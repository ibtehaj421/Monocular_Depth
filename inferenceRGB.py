import torch
import cv2
import numpy as np
import pandas as pd
import os
import argparse
from torchvision import transforms
from modelSmol import DepthModel

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
    print(f"Loading model from {checkpoint_path}...")
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

# --- INTERACTIVE LOGIC ---
class InteractiveState:
    def __init__(self, rgb_img, pred_map, gt_map=None):
        self.original_rgb = rgb_img.copy()
        self.pred_map = pred_map
        self.gt_map = gt_map
        self.points = []
        self.mode = 'rgb' 
        
        # Colorize predictions for display
        pred_norm = cv2.normalize(pred_map, None, 0, 255, cv2.NORM_MINMAX)
        self.pred_viz = cv2.applyColorMap(pred_norm.astype(np.uint8), cv2.COLORMAP_INFERNO)
        
        # Colorize GT for display (if exists)
        if gt_map is not None:
            gt_norm = cv2.normalize(gt_map, None, 0, 255, cv2.NORM_MINMAX)
            self.gt_viz = cv2.applyColorMap(gt_norm.astype(np.uint8), cv2.COLORMAP_INFERNO)
        else:
            self.gt_viz = np.zeros_like(self.pred_viz)

        self.current_img = self.original_rgb.copy()

    def toggle_mode(self):
        if self.mode == 'rgb': self.mode = 'pred'
        elif self.mode == 'pred': self.mode = 'gt'
        else: self.mode = 'rgb'
        self.refresh_display()

    def refresh_display(self):
        if self.mode == 'rgb': base = self.original_rgb
        elif self.mode == 'pred': base = self.pred_viz
        else: base = self.gt_viz
        
        self.current_img = base.copy()
        
        # Draw overlay text for mode
        cv2.putText(self.current_img, f"MODE: {self.mode.upper()}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        for i, p in enumerate(self.points):
            cv2.circle(self.current_img, p, 5, (0, 255, 0), -1)
            if i > 0:
                cv2.line(self.current_img, self.points[i-1], p, (0, 255, 255), 2)
        
        if len(self.points) == 2:
            self.calc_and_draw(self.points[0], self.points[1])

        cv2.imshow("Analysis Tool", self.current_img)

    def calc_and_draw(self, p1, p2):
        # 1. Prediction Math
        h, w = self.pred_map.shape
        # Clamp coordinates
        p1 = (min(p1[0], w-1), min(p1[1], h-1))
        p2 = (min(p2[0], w-1), min(p2[1], h-1))
        
        d1_p = self.pred_map[p1[1], p1[0]]
        d2_p = self.pred_map[p2[1], p2[0]]
        
        P1_3d = pixel_to_3d(p1[0], p1[1], d1_p, K_MATRIX)
        P2_3d = pixel_to_3d(p2[0], p2[1], d2_p, K_MATRIX)
        dist_pred = np.linalg.norm(P1_3d - P2_3d)
        
        # 2. Ground Truth Math (if available)
        dist_gt = -1.0
        if self.gt_map is not None:
            d1_g = self.gt_map[p1[1], p1[0]]
            d2_g = self.gt_map[p2[1], p2[0]]
            
            P1_gt = pixel_to_3d(p1[0], p1[1], d1_g, K_MATRIX)
            P2_gt = pixel_to_3d(p2[0], p2[1], d2_g, K_MATRIX)
            dist_gt = np.linalg.norm(P1_gt - P2_gt)
            
            # Error Calc
            error_m = abs(dist_pred - dist_gt)
            error_pct = (error_m / (dist_gt + 1e-6)) * 100

        # 3. Draw on Screen
        label = f"Pred: {dist_pred:.2f}m"
        cv2.putText(self.current_img, label, (p1[0], p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 4. Print Detailed Stats to Terminal
        print(f"\n--- MEASUREMENT ANALYSIS ---")
        print(f"Pixels: {p1} -> {p2}")
        print(f"[PREDICTION] Dist: {dist_pred:.4f}m | Depths: {d1_p:.2f}m -> {d2_p:.2f}m")
        
        if dist_gt != -1.0:
            print(f"[GROUND TRUTH] Dist: {dist_gt:.4f}m | Depths: {d1_g:.2f}m -> {d2_g:.2f}m")
            print(f"[ACCURACY] Error: {error_m:.4f}m ({error_pct:.1f}%)")
            
            # Visual feedback on screen for GT
            label_gt = f"GT: {dist_gt:.2f}m"
            cv2.putText(self.current_img, label_gt, (p1[0], p1[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            print("[GROUND TRUTH] Not Available for this image.")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) >= 2: self.points = []
            self.points.append((x, y))
            self.refresh_display()

def run_inference(model_path, image_path=None):
    model = load_model(model_path)
    gt_map = None
    
    # 1. Select Image & GT
    if image_path is None:
        # Load from Validation Split
        df = pd.read_csv('data/val_split.csv', header=None)
        row = df.sample(1).iloc[0]
        image_path = "data/nyu2_test/00059_colors.png"
        gt_path = "data/nyu2_test/00059_depth.png"
        
        if os.path.exists(gt_path):
            gt_raw = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            # Smart Scale Logic
            if gt_raw.max() > 255: gt_map = gt_raw.astype('float32') / 1000.0
            else: gt_map = gt_raw.astype('float32') / 255.0 * 10.0
            gt_map = cv2.resize(gt_map, (320, 320), interpolation=cv2.INTER_NEAREST)

    print(f"Running on: {image_path}")
    
    # 2. Preprocess
    img_bgr = cv2.imread(image_path)
    img_bgr_320 = cv2.resize(img_bgr, (320, 320))
    img_rgb = cv2.cvtColor(img_bgr_320, cv2.COLOR_BGR2RGB)
    img_tensor = transforms.ToTensor()(img_rgb)
    img_tensor = NORMALIZE(img_tensor).unsqueeze(0).to(DEVICE)
    
    # 3. Predict
    with torch.no_grad():
        pred_map = model(img_tensor).squeeze().cpu().numpy()

    # 4. Interactive Mode
    print("\n>>> ANALYSIS TOOL <<<")
    print(" [Click] : Measure Distance")
    print(" [T Key] : Cycle Views (RGB -> PRED -> GT)")
    print(" [Q Key] : Quit")
    
    state = InteractiveState(img_bgr_320, pred_map, gt_map)
    cv2.imshow("Analysis Tool", state.current_img)
    cv2.setMouseCallback("Analysis Tool", state.mouse_callback)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('t'): state.toggle_mode()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SecondPass/checkpoint_chunk_5_pass2.pth')
    parser.add_argument('--image', type=str, default=None)
    args = parser.parse_args()
    
    # Find best backup model if default missing
    if not os.path.exists(args.model):
        for backup in ['checkpoints/checkpoint_final_chunk_6.pth', 'checkpoints/checkpoint_chunk_5_pass2.pth', 'checkpoints/checkpoint_latest.pth']:
            if os.path.exists(backup):
                print(f"Using backup: {backup}")
                args.model = backup
                break
                
    run_inference(args.model, args.image)