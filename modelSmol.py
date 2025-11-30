import torch
import torch.nn as nn
import torchvision.models as models

class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()        
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.bn = nn.BatchNorm2d(output_features)

    def forward(self, x, concat_with):
        # Upsample input
        x = torch.nn.functional.interpolate(x, size=[concat_with.shape[2], concat_with.shape[3]], mode='bilinear', align_corners=True)
        # Concatenate with skip connection
        return self.convA(torch.cat([x, concat_with], dim=1))

class DepthModel(nn.Module):
    def __init__(self):
        super(DepthModel, self).__init__()
        # 1. Encoder (ResNet18)
        self.encoder = models.resnet18(pretrained=True)
        
        # 2. Decoder (Upsampling layers)
        # These channel numbers (512, 256, etc.) match ResNet18's layer outputs
        self.up1 = UpSample(skip_input=512 + 256, output_features=256)
        self.up2 = UpSample(skip_input=256 + 128, output_features=128)
        self.up3 = UpSample(skip_input=128 + 64,  output_features=64)
        self.up4 = UpSample(skip_input=64 + 64,   output_features=32)
        
        # 3. Final Output Layer (1 Channel for Depth)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder Pass (Save features for skip connections)
        x0 = self.encoder.conv1(x)
        x0 = self.encoder.bn1(x0)
        x0 = self.encoder.relu(x0)
        x1 = self.encoder.maxpool(x0) # 64 channels
        
        x2 = self.encoder.layer1(x1)  # 64 channels
        x3 = self.encoder.layer2(x2)  # 128 channels
        x4 = self.encoder.layer3(x3)  # 256 channels
        x5 = self.encoder.layer4(x4)  # 512 channels (Bottleneck)
        
        # Decoder Pass (with Skip Connections)
        d5 = self.up1(x5, x4)
        d4 = self.up2(d5, x3)
        d3 = self.up3(d4, x2)
        d2 = self.up4(d3, x0) # Skip connection to early layer
        
        # Final Output
        out = self.final_conv(d2)
        
        # Resize to original input size (if needed)
        out = torch.nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        return self.sigmoid(out) * 10.0 # Scale 0-1 output to 0-10 meters