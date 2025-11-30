# Monocular Metric Depth Estimation using ResNet-34

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![CUDA](https://img.shields.io/badge/CUDA-11.8-green)

A deep learning system for estimating metric depth from single RGB images using a U-Net architecture with ResNet-34 encoder and Curriculum Learning strategy.

## 🎯 Project Overview

This project implements a robust monocular depth estimation pipeline optimized for consumer hardware (RTX 3060). The system achieves **0.37m Mean Absolute Error** on the NYU Depth V2 benchmark with relative errors as low as **2.3%** for near-field objects.

### Key Features

- **U-Net Architecture** with pre-trained ResNet-34 encoder
- **Curriculum Learning** strategy for memory-efficient training
- **Edge-Aware Loss** function for sharp boundary preservation
- **Metric Depth Prediction** (absolute scale in meters)
- Optimized for **consumer-grade GPUs** (12GB VRAM)

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.10+
PyTorch 2.0+
CUDA 11.8+
NVIDIA GPU with 12GB+ VRAM (recommended)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/ibtehaj421/Monocular_Depth.git
cd depth-estimation

# Install dependencies
pip install torch torchvision opencv-python numpy matplotlib pillow

# Download NYU Depth V2 dataset
# Place in ./data/nyu_depth_v2/
```

### Training

```bash
# Phase 1: Base training with curriculum learning
python train.py --phase 1 --lr 1e-4 --epochs 50

# Phase 2: Fine-tuning with reduced learning rate
python train.py --phase 2 --lr 1e-5 --epochs 30 --checkpoint phase1_final.pth
```

### Inference

```bash
# Run inference on a single image
python inference_r34.py --image path/to/image.jpg --model checkpoints/resnet34_final.pth

# Batch inference on test set
python evaluate.py --test_dir data/nyu_test --model checkpoints/resnet34_final.pth
```

## 📊 Results

| Model Configuration | MAE (m) | RMSE (m) | δ < 1.25 (%) |
|---------------------|---------|----------|--------------|
| ResNet-18 Baseline  | 0.44    | 0.58     | 81.2         |
| **ResNet-34 (Final)** | **0.37** | **0.49** | **87.3**     |

### Accuracy by Range

- **Close Range (<1m):** 2.3% relative error
- **Medium Range (1-3m):** 5-10% relative error  
- **Far Range (>5m):** 20-25% relative error

## 🏗️ Architecture

The system uses a U-Net encoder-decoder architecture:

- **Encoder:** ResNet-34 (pre-trained on ImageNet)
- **Decoder:** Transposed convolutions with skip connections
- **Input:** 320×320 RGB images
- **Output:** 320×320 depth maps (0-10m range)

### Loss Function

```python
L_total = L_depth + λ(L_grad_x + L_grad_y)
```

Where:

- `L_depth`: L1 loss for pixel-wise depth accuracy
- `L_grad`: Gradient loss for edge preservation
- `λ = 0.5`: Balancing parameter

## 📁 Project Structure

```

depth-estimation/
├── train.py                 # Training script with curriculum learning
├── inference_r34.py         # Single image inference
├── evaluate.py              # Batch evaluation on test set
├── model.py                 # U-Net + ResNet-34 architecture
├── loss.py                  # Edge-aware loss implementation
├── data_loader.py           # NYU Depth V2 dataset loader
├── checkpoints/             # Saved model weights
├── data/                    # Dataset directory
│   └── nyu_depth_v2/
└── results/                 # Output visualizations

```

## 🔧 Key Implementation Details

### Curriculum Learning

- Dataset split into 10 chunks (~5,000 images each)
- Sequential training prevents memory overflow
- Two-phase approach: base training + fine-tuning

### Data Preprocessing Fix

Critical issue resolved: NYU Depth V2 has inconsistent encoding

- Training images: 8-bit (0-255) → scale by `/255 * 10`

- Test images: 16-bit (0-65535) millimeters → scale by `/1000`

### Hyperparameters

- Batch size: 16
- Learning rate: 1e-4 (Phase 1), 1e-5 (Phase 2)
- Optimizer: Adam
- Weight decay: 1e-4
- Input resolution: 320×320

## 🎓 Applications

- **Robotics:** Obstacle avoidance and navigation
- **Augmented Reality:** Realistic object occlusion
- **3D Reconstruction:** Indoor scene mapping
- **Assistive Technology:** Visual aids for navigation

## 📝 Citation

```bibtex
@project{haider2024depth,
  title={Monocular Metric Depth Estimation using ResNet-34 and Curriculum Learning},
  author={Haider, Ibtehaj},
  year={2024},
  institution={FAST NUCES Islamabad}
}
```

## 🙏 Acknowledgments

- NYU Depth V2 Dataset: Silberman et al. (2012)
- ResNet Architecture: He et al. (2016)
- U-Net Design: Ronneberger et al. (2015)
