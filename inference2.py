import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from torchvision import transforms
from modelSmol import DepthModel

# --- Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Standard ImageNet normalization
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

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

def preprocess_image(image_path, target_size=(320, 320)):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)
    img_tensor = transforms.ToTensor()(img_resized)
    img_tensor = NORMALIZE(img_tensor).unsqueeze(0).to(DEVICE)
    return img_bgr, img_resized, img_tensor

def pixel_to_3d(u, v, depth_val, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) * depth_val / fx
    y = (v - cy) * depth_val / fy
    z = depth_val
    return np.array([x, y, z])

def interactive_distance_measure(event, x, y, flags, param):
    depth_map, display_img = param
    if event == cv2.EVENT_LBUTTONDOWN:
        if not hasattr(interactive_distance_measure, 'points'):
            interactive_distance_measure.points = []
        interactive_distance_measure.points.append((x, y))
        cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Distance Measurement", display_img)
        
        if len(interactive_distance_measure.points) == 2:
            p1 = interactive_distance_measure.points[0]
            p2 = interactive_distance_measure.points[1]
            h, w = depth_map.shape
            d1 = depth_map[min(p1[1], h-1), min(p1[0], w-1)]
            d2 = depth_map[min(p2[1], h-1), min(p2[0], w-1)]
            P1_3d = pixel_to_3d(p1[0], p1[1], d1, K_MATRIX)
            P2_3d = pixel_to_3d(p2[0], p2[1], d2, K_MATRIX)
            dist = np.linalg.norm(P1_3d - P2_3d)
            
            print(f"Distance: {dist:.4f} meters")
            cv2.line(display_img, p1, p2, (255, 0, 0), 2)
            cv2.putText(display_img, f"{dist:.2f}m", (p1[0], p1[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Distance Measurement", display_img)
            interactive_distance_measure.points = []

def run_inference(model_path, image_path=None):
    model = load_model(model_path)
    
    # Select Image
    gt_depth = None
    if image_path is None:
        df = pd.read_csv('data/val_split.csv', header=None)
        row = df.sample(1).iloc[0]
        image_path = "data/nyu2_test/00059_colors.png"
        gt_path = "data/nyu2_test/00059_depth.png"
        
        if os.path.exists(gt_path):
            gt_raw = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            # Smart Scaling Logic
            if gt_raw.max() > 255:
                 gt_depth = gt_raw.astype('float32') / 1000.0
            else:
                 gt_depth = gt_raw.astype('float32') / 255.0 * 10.0
            gt_depth = cv2.resize(gt_depth, (320, 320), interpolation=cv2.INTER_NEAREST)

    print(f"Running inference on: {image_path}")
    _, rgb_resized, input_tensor = preprocess_image(image_path)
    
    with torch.no_grad():
        pred_depth = model(input_tensor).squeeze().cpu().numpy()

    # --- VISUALIZATION WITH DIFFERENCE MAP ---
    plt.figure(figsize=(20, 5))
    
    # 1. RGB
    plt.subplot(1, 4, 1)
    plt.title("Input RGB")
    plt.imshow(rgb_resized)
    plt.axis('off')
    
    # 2. Prediction
    plt.subplot(1, 4, 2)
    plt.title("Predicted Depth")
    plt.imshow(pred_depth, cmap='inferno', vmin=0, vmax=10)
    plt.colorbar(label='Meters')
    plt.axis('off')

    # 3. Ground Truth
    if gt_depth is not None:
        plt.subplot(1, 4, 3)
        plt.title("Ground Truth")
        plt.imshow(gt_depth, cmap='inferno', vmin=0, vmax=10)
        plt.colorbar(label='Meters')
        plt.axis('off')

        # 4. Difference Map (Error)
        plt.subplot(1, 4, 4)
        
        # Calculate Absolute Difference
        diff = np.abs(gt_depth - pred_depth)
        mae = np.mean(diff)
        
        plt.title(f"Error Map (MAE: {mae:.3f}m)")
        # 'magma' or 'hot' highlights errors brightly
        plt.imshow(diff, cmap='magma', vmin=0, vmax=2.0) 
        plt.colorbar(label='Error (Meters)')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    # Interactive Measurement
    depth_norm = cv2.normalize(pred_depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_color = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.imshow("Distance Measurement", depth_color)
    cv2.setMouseCallback("Distance Measurement", interactive_distance_measure, [pred_depth, depth_color])
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SecondPass/checkpoint_chunk_5_pass2.pth', help='Path to .pth model')
    parser.add_argument('--image', type=str, default=None, help='Path to specific image')
    args = parser.parse_args()
    
    # Fallback to older checkpoint if pass2 doesn't exist yet
    if not os.path.exists(args.model) and os.path.exists('checkpoints/checkpoint_latest.pth'):
        args.model = 'checkpoints/checkpoint_latest.pth'
        
    run_inference(args.model, args.image)