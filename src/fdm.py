import torch
import torch.nn as nn
import torch.nn.functional as F


class FDM(nn.Module):
    def __init__(self, in_channels, filters):
        super(FDM, self).__init__()

        # Initial convolution layer with fewer filters
        self.conv1 = nn.Conv2d(in_channels, filters,
                               kernel_size=3, stride=2, padding=1)

        # Using average pooling for downsampling
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # Depthwise separable convolution to reduce parameters
        self.depthwise_conv = nn.Conv2d(
            in_channels + filters, in_channels + filters, kernel_size=3, stride=1, padding=1, groups=in_channels + filters
        )

        # Pointwise convolution to reduce dimensionality
        self.pointwise_conv = nn.Conv2d(
            in_channels + filters, filters, kernel_size=1
        )

        # Channel transformation layer with fewer filters
        self.channel_transform = nn.Conv2d(
            filters, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = self.avgpool(x)

        x3 = torch.cat([x1, x2], dim=1)
        x4 = F.relu(self.depthwise_conv(x3))
        x5 = F.relu(self.pointwise_conv(x4))

        # Channel transformation to match input channels
        x6 = F.relu(self.channel_transform(x5))

        # Spatial Detail Recovery Process (SDRP)
        x7 = F.interpolate(x6, scale_factor=2,
                           mode='bilinear', align_corners=False)

        return x - x7, x5
