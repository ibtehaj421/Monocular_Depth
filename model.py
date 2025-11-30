
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
        # Upsample input to match skip connection size
        x = torch.nn.functional.interpolate(x, size=[concat_with.shape[2], concat_with.shape[3]], mode='bilinear', align_corners=True)
        return self.convA(torch.cat([x, concat_with], dim=1))

class DepthModel(nn.Module):
    def __init__(self):
        super(DepthModel, self).__init__()
        
        # --- UPGRADE: Use ResNet-34 ---
        # It has more layers than ResNet-18 but same channel dimensions [64, 128, 256, 512]
        self.encoder = models.resnet34(weights='IMAGENET1K_V1')
        
        # Decoder logic remains identical to ResNet-18 because channel sizes match
        self.up1 = UpSample(skip_input=512 + 256, output_features=256)
        self.up2 = UpSample(skip_input=256 + 128, output_features=128)
        self.up3 = UpSample(skip_input=128 + 64,  output_features=64)
        self.up4 = UpSample(skip_input=64 + 64,   output_features=32)
        
        self.final_conv = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x0 = self.encoder.conv1(x)
        x0 = self.encoder.bn1(x0)
        x0 = self.encoder.relu(x0)
        x1 = self.encoder.maxpool(x0) # 64
        
        x2 = self.encoder.layer1(x1)  # 64
        x3 = self.encoder.layer2(x2)  # 128
        x4 = self.encoder.layer3(x3)  # 256
        x5 = self.encoder.layer4(x4)  # 512
        
        # Decoder
        d5 = self.up1(x5, x4)
        d4 = self.up2(d5, x3)
        d3 = self.up3(d4, x2)
        d2 = self.up4(d3, x0)
        
        out = self.final_conv(d2)
        out = torch.nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        return self.sigmoid(out) * 10.0 # Scale 0-10m