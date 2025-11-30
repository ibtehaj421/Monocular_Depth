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
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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

def compute_edge_map(depth_map):
    """
    Computes the gradient magnitude (edge strength) of a depth map.
    Returns: Edge Map (Unsigned Float) where high values = sharp depth jumps.
    """
    # 1. Calculate Gradients in X and Y using Sobel
    # ksize=3 is standard for edge detection
    grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
    
    # 2. Magnitude = sqrt(x^2 + y^2)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    return magnitude

def visualize_edge_analysis(rgb, pred, gt):
    # 1. Compute Edges
    pred_edges = compute_edge_map(pred)
    gt_edges = compute_edge_map(gt)
    
    # 2. Compute Edge Difference (Where do they disagree?)
    # Red/Hot = Model missed an edge or hallucinated one
    edge_diff = np.abs(pred_edges - gt_edges)
    
    # 3. Metrics
    # Mean Gradient: How "sharp" is the image on average?
    # If GT > Pred, the model is too smooth/blurry.
    sharpness_pred = np.mean(pred_edges)
    sharpness_gt = np.mean(gt_edges)
    sharpness_ratio = (sharpness_pred / (sharpness_gt + 1e-6)) * 100
    
    print(f"\n--- EDGE METRICS ---")
    print(f"Avg Edge Strength (Pred): {sharpness_pred:.4f}")
    print(f"Avg Edge Strength (Real): {sharpness_gt:.4f}")
    print(f"Sharpness Ratio: {sharpness_ratio:.1f}% (100% = Perfect Match)")
    
    # 4. Plotting
    plt.figure(figsize=(18, 10))
    
    # Row 1: Depth Maps
    plt.subplot(2, 3, 1)
    plt.title("Input RGB")
    plt.imshow(rgb)
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.title("Predicted Depth")
    plt.imshow(pred, cmap='inferno', vmin=0, vmax=10)
    plt.colorbar(label='Meters')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.title("Ground Truth Depth")
    plt.imshow(gt, cmap='inferno', vmin=0, vmax=10)
    plt.colorbar(label='Meters')
    plt.axis('off')
    
    # Row 2: Edge Maps
    # We use a threshold for visualization to make edges pop
    vmax_edge = 2.0 # Cap gradients for visualization
    
    plt.subplot(2, 3, 4)
    plt.title(f"Predicted Edges\n(Sharpness: {sharpness_pred:.3f})")
    plt.imshow(pred_edges, cmap='gray', vmin=0, vmax=vmax_edge)
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.title(f"Ground Truth Edges\n(Sharpness: {sharpness_gt:.3f})")
    plt.imshow(gt_edges, cmap='gray', vmin=0, vmax=vmax_edge)
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.title("Edge Error Map\n(Brighter = Mismatch)")
    plt.imshow(edge_diff, cmap='hot', vmin=0, vmax=2.0)
    plt.colorbar(label='Gradient Diff')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def run_analysis(model_path, image_path=None):
    model = load_model(model_path)
    
    # Select Image
    if image_path is None:
        if os.path.exists('data/nyu2_test.csv'):
            df = pd.read_csv('data/nyu2_test.csv', header=None)
        else:
            df = pd.read_csv('data/val_split.csv', header=None)
        row = df.sample(1).iloc[0]
        image_path = row[0]
        gt_path = row[1]
    
    print(f"Analyzing: {image_path}")
    
    # Load & Preprocess
    img_bgr = cv2.imread(image_path)
    if img_bgr is None: raise FileNotFoundError(f"Missing {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (320, 320))
    
    tensor = transforms.ToTensor()(img_resized)
    tensor = NORMALIZE(tensor).unsqueeze(0).to(DEVICE)
    
    # Load GT
    gt_raw = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    if gt_raw.max() > 255: gt = gt_raw.astype('float32') / 1000.0
    else: gt = gt_raw.astype('float32') / 255.0 * 10.0
    gt = cv2.resize(gt, (320, 320), interpolation=cv2.INTER_NEAREST)
    
    # Predict
    with torch.no_grad():
        pred = model(tensor).squeeze().cpu().numpy()
        
    # Run Visualizer
    visualize_edge_analysis(img_resized, pred, gt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Auto-find best model
    default_model = 'checkpoints_r34_finetune/finetuned_chunk_5.pth'
    if not os.path.exists(default_model):
        for i in range(9, -1, -1):
            p = f'checkpoints_r34/checkpoint_r34_chunk_{i}.pth'
            if os.path.exists(p): 
                default_model = p
                break
                
    parser.add_argument('--model', type=str, default=default_model)
    parser.add_argument('--image', type=str, default=None)
    args = parser.parse_args()
    
    run_analysis(args.model, args.image)