import torch
import torch.nn as nn

class MaxFeatureMap2D(nn.Module):
    """
    Max-Feature-Map activation
    Key component of LCNN that reduces feature maps by taking max over pairs
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # x shape: (batch, channels, ...)
        # We assume channels is the second dimension and should be halved
        shape = list(x.size())
        batch_size = shape[0]
        channels = shape[1]
        
        # Split channels into pairs and take max
        # New shape: (batch, channels // 2, 2, ...)
        new_shape = [batch_size, channels // 2, 2] + shape[2:]
        x = x.view(*new_shape)
        x, _ = torch.max(x, dim=2)
        return x


class MFMConv2d(nn.Module):
    """Convolutional layer with Max-Feature-Map activation"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels * 2,  # Double channels for MFM
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.mfm = MaxFeatureMap2D()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.mfm(x)
        return x


class LCNN(nn.Module):
    """
    Light CNN for audio deepfake detection
    Optimized for LFCC features
    """
    def __init__(self, n_lfcc=60, num_classes=2):
        super().__init__()
        
        # Input: (batch, 1, n_lfcc, time)
        self.conv1 = MFMConv2d(1, 48, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2a = MFMConv2d(48, 48, kernel_size=1, stride=1, padding=0)
        self.conv2 = MFMConv2d(48, 96, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3a = MFMConv2d(96, 96, kernel_size=1, stride=1, padding=0)
        self.conv3 = MFMConv2d(96, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4a = MFMConv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.conv4 = MFMConv2d(128, 192, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5a = MFMConv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.conv5 = MFMConv2d(192, 256, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers with MFM
        self.fc1 = nn.Linear(256, 320)
        self.mfm_fc1 = MaxFeatureMap2D()
        self.dropout1 = nn.Dropout(0.7)
        
        self.fc2 = nn.Linear(160, num_classes)
        
    def forward(self, x):
        # x: (batch, 1, n_lfcc, time)
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2a(x)
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.conv3a(x)
        x = self.conv3(x)
        x = self.pool3(x)
        
        x = self.conv4a(x)
        x = self.conv4(x)
        x = self.pool4(x)
        
        x = self.conv5a(x)
        x = self.conv5(x)
        x = self.pool5(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.fc1(x)
        x = self.mfm_fc1(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        return x
