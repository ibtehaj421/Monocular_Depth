import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class NYUDepthDataset(Dataset):
    def __init__(self, csv_file, root_dir, target_size=(320, 320), is_train=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            target_size (tuple): Desired output size (height, width).
            is_train (bool): If true, applies data augmentation.
        """
        self.data_frame = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.target_size = target_size
        self.is_train = is_train
        
        # Standard ImageNet normalization for the ResNet backbone
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 1. Parse Paths
        rgb_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        depth_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 1])

        # 2. Load Images
        image = cv2.imread(rgb_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load Depth (Crucial: IMREAD_UNCHANGED to detect 16-bit vs 8-bit)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) 

        # 3. Preprocessing & Resizing
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, self.target_size, interpolation=cv2.INTER_NEAREST)

        # 4. SMART DEPTH NORMALIZATION
        # Check the data type to decide how to normalize
        depth = depth.astype('float32')
        
        if depth.max() > 255.0:
            # Case A: It's 16-bit Millimeters (e.g., 1798)
            # 1798 mm -> 1.798 meters
            depth = depth / 1000.0 
        else:
            # Case B: It's 8-bit Quantized (e.g., 254)
            # 255 -> 10.0 meters (Approximation)
            depth = depth / 255.0 * 10.0

        # Cap max depth to 10 meters (standard for NYU) to remove noise/infinity
        depth[depth > 10.0] = 10.0

        # 5. To Tensor
        image = transforms.ToTensor()(image)
        image = self.norm(image)
        
        depth = torch.from_numpy(depth).unsqueeze(0)

        return image, depth

# Usage Example:
# train_dataset = NYUDepthDataset(csv_file='nyu2_train.csv', root_dir='./')
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)