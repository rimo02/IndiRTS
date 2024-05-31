import torch
import torch.nn as nn
import torch.nn.functional as F
from .global_feature import GlobalFeature


class DLM1(nn.Module):
    def __init__(self, c, k):
        super(DLM1, self).__init__()
        self.c1 = nn.Conv2d(in_channels=64, out_channels=64,
                            kernel_size=3, stride=2, padding=1)
        self.globalfeature = GlobalFeature(c, k)
        self.maxpool = nn.MaxPool2d(kernel_size=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2)

    def forward(self, x):
        x1 = F.relu(self.c1(x))
        x2, indices = self.maxpool(x)
        x3 = x1 + x2
        x4 = self.globalfeature(x3)
        x5 = x3 - x4
        x6 = F.interpolate(x5, scale_factor=2,
                           mode='bilinear', align_corners=False)
        x7 = self.unpool(x5, indices)
        x8 = x6 + x7
        return x8, x4


class DLM2(nn.Module):
    def __init__(self, c, k):
        super(DLM2, self).__init__()
        self.c1 = nn.Conv2d(in_channels=64, out_channels=64,
                            kernel_size=3, stride=2, padding=1)
        self.globalfeature = GlobalFeature(c, k)
        self.maxpool = nn.MaxPool2d(kernel_size=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2)

    def forward(self, x):
        x1 = F.relu(self.c1(x))
        x2, indices = self.maxpool(x)
        x3 = x1 + x2
        x4 = self.globalfeature(x3)
        x4 = self.globalfeature(x4)
        x5 = x3 - x4
        x6 = F.interpolate(x5, scale_factor=2,
                           mode='bilinear', align_corners=False)
        x7 = self.unpool(x5, indices)
        x8 = x6 + x7
        return x8, x4
