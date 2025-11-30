import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from torchvision import transforms
from model import DepthModel

# --- Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Standard ImageNet normalization (Must match training!)
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Approximate Kinect Intrinsics (Scaled to 320x320)
# Original (640x480): fx=518.85, fy=518.85, cx=325.58, cy=253.74
# Scale factors: x * (320/640) = 0.5, y * (320/480) = 0.666
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
        
        # Handle state dicts saved inside a wrapper dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        print("Model loaded successfully.")
        return model
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

def preprocess_image(image_path, target_size=(320, 320)):
    # Load Image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize
    img_resized = cv2.resize(img_rgb, target_size)
    
    # Normalize & Tensor
    img_tensor = transforms.ToTensor()(img_resized) # 0-1
    img_tensor = NORMALIZE(img_tensor).unsqueeze(0).to(DEVICE) # Add batch dim
    
    return img_bgr, img_resized, img_tensor

def pixel_to_3d(u, v, depth_val, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) * depth_val / fx
    y = (v - cy) * depth_val / fy
    z = depth_val
    return np.array([x, y, z])

def interactive_distance_measure(event, x, y, flags, param):
    """
    Mouse callback function to measure distance between two clicks.
    param: (depth_map_in_meters, original_image_for_display)
    """
    depth_map, display_img = param
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if we have global points list
        if not hasattr(interactive_distance_measure, 'points'):
            interactive_distance_measure.points = []

        # Add point
        interactive_distance_measure.points.append((x, y))
        cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Distance Measurement", display_img)
        
        if len(interactive_distance_measure.points) == 2:
            p1 = interactive_distance_measure.points[0]
            p2 = interactive_distance_measure.points[1]
            
            # Get depth (Handle out of bounds safely)
            h, w = depth_map.shape
            d1 = depth_map[min(p1[1], h-1), min(p1[0], w-1)]
            d2 = depth_map[min(p2[1], h-1), min(p2[0], w-1)]
            
            # Convert to 3D
            P1_3d = pixel_to_3d(p1[0], p1[1], d1, K_MATRIX)
            P2_3d = pixel_to_3d(p2[0], p2[1], d2, K_MATRIX)
            
            # Calculate Distance
            dist = np.linalg.norm(P1_3d - P2_3d)
            
            print(f"\n--- Measurement ---")
            print(f"Point 1: {p1} | Depth: {d1:.2f}m")
            print(f"Point 2: {p2} | Depth: {d2:.2f}m")
            print(f"Euclidean Distance: {dist:.4f} meters")
            
            # Draw line
            cv2.line(display_img, p1, p2, (255, 0, 0), 2)
            cv2.putText(display_img, f"{dist:.2f}m", (p1[0], p1[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Distance Measurement", display_img)
            
            # Reset
            interactive_distance_measure.points = []

def run_inference(model_path, image_path=None):
    # 1. Load Model
    model = load_model(model_path)
    
    # 2. Select Image
    gt_depth = None
    if image_path is None:
        # Randomly pick from validation split
        df = pd.read_csv('data/val_split.csv', header=None)
        row = df.sample(1).iloc[0]
        image_path = "data/nyu2_test/00059_colors.png"
        gt_path ="data/nyu2_test/00059_depth.png"
        
        # Load Ground Truth for comparison
        if os.path.exists(gt_path):
             # Load raw (could be 16-bit or 8-bit)
            gt_raw = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            
            # --- SMART DEPTH SCALING (UPDATED) ---
            # Automatically detect format
            if gt_raw.max() > 255:
                 # 16-bit mm (Divide by 1000 to get meters)
                 gt_depth = gt_raw.astype('float32') / 1000.0
            else:
                 # 8-bit Quantized (Scale 0-255 -> 0-10 meters)
                 gt_depth = gt_raw.astype('float32') / 255.0 * 10.0
            
            gt_depth = cv2.resize(gt_depth, (320, 320), interpolation=cv2.INTER_NEAREST)

    print(f"Running inference on: {image_path}")
    
    # 3. Preprocess
    orig_bgr, rgb_resized, input_tensor = preprocess_image(image_path)
    
    # 4. Predict
    with torch.no_grad():
        pred_depth = model(input_tensor)
        # Convert tensor to numpy array (H, W)
        pred_depth = pred_depth.squeeze().cpu().numpy()

    # 5. Visualization (Matplotlib - Static Comparison)
    plt.figure(figsize=(15, 5))
    
    # RGB
    plt.subplot(1, 3, 1)
    plt.title("Input RGB")
    plt.imshow(rgb_resized)
    plt.axis('off')
    
    # Prediction (Colorized)
    plt.subplot(1, 3, 2)
    plt.title("Predicted Depth")
    # vmin/vmax ensures colorbar is locked to 0-10m for consistency
    plt.imshow(pred_depth, cmap='inferno', vmin=0, vmax=10) 
    plt.colorbar(label='Depth (m)')
    plt.axis('off')

    # Ground Truth (if available)
    if gt_depth is not None:
        plt.subplot(1, 3, 3)
        plt.title("Ground Truth")
        plt.imshow(gt_depth, cmap='inferno', vmin=0, vmax=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    # 6. Interactive Measurement (OpenCV)
    print("\nOpening Interactive Window...")
    print(">> CLICK 2 POINTS on the image to measure distance.")
    print(">> Press 'q' to exit.")
    
    # Prepare display image (Prediction colorized)
    # Normalize prediction to 0-255 for visualization
    depth_norm = cv2.normalize(pred_depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_color = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_INFERNO)
    
    cv2.imshow("Distance Measurement", depth_color)
    cv2.setMouseCallback("Distance Measurement", interactive_distance_measure, [pred_depth, depth_color])
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Defaults to checkpoint_chunk_5.pth as requested for final model
    parser.add_argument('--model', type=str, default='SecondPass/checkpoint_chunk_5_pass2.pth', help='Path to .pth model')
    parser.add_argument('--image', type=str, default=None, help='Path to specific image (optional)')
    args = parser.parse_args()
    
    # Fallback if chunk 5 doesn't exist yet (e.g. still training)
    if not os.path.exists(args.model) and os.path.exists('checkpoints/checkpoint_latest.pth'):
        print(f"Note: {args.model} not found. Using checkpoint_latest.pth instead.")
        args.model = 'checkpoints/checkpoint_latest.pth'
    
    run_inference(args.model, args.image)