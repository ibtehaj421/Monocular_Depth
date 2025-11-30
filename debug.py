import cv2
import numpy as np
import pandas as pd
import os

# Read one path from your CSV
df = pd.read_csv('data/nyu2_train.csv', header=None)
depth_path = os.path.join('.', df.iloc[0, 1]) # Adjust 'data' if needed

# 1. Load EXACTLY how the training loop does
depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

if depth_raw is None:
    print("Error: Could not find image. Check paths.")
else:
    print(f"--- Data Inspection ---")
    print(f"Image Type: {depth_raw.dtype}")
    print(f"Raw Min Value: {depth_raw.min()}")
    print(f"Raw Max Value: {depth_raw.max()}")
    
    # Simulate the preprocessing
    depth_processed = depth_raw.astype('float32') / 1000.0
    
    print(f"Processed Max Value (What the model sees): {depth_processed.max()}")
    
    if depth_processed.max() < 1.0:
        print("\n[CRITICAL WARNING]: Your max depth is less than 1 meter.")
        print("The /1000.0 scaling is incorrect for your dataset.")
        print("Your model is training on 'microscopic' depth maps.")
    elif depth_processed.max() > 100.0:
        print("\n[WARNING]: Your max depth is huge (>100m).")
        print("You might have missed the scaling factor entirely.")
    else:
        print("\n[OK]: Data looks reasonable (1m - 10m range).")