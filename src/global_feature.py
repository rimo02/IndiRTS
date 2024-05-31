import torch
import torch.nn as nn
import torch.nn.functional as F
from .bottleneck import Bottleneck


class GlobalFeature(nn.Module):
    def __init__(self, c, growth_rate):
        super(GlobalFeature, self).__init__()
        self.initial_conv = nn.Conv2d(
            in_channels=64, out_channels=c, kernel_size=3, stride=1, padding=1)

        # First bottleneck block
        self.bottleneck1 = Bottleneck(c, c // 4)

        # Second block with depthwise separable convolution
        self.depthwise_separable = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c),
            nn.Conv2d(c, c + growth_rate, kernel_size=1)
        )

        # Third block with dilated convolution
        self.dilated_conv = nn.Conv2d(
            c + growth_rate, c + 2 * growth_rate, kernel_size=3, stride=1, padding=2, dilation=2)

        # Fourth block with standard convolution and skip connection
        self.conv_skip = nn.Conv2d(
            c + 2 * growth_rate, c + 3 * growth_rate, kernel_size=3, stride=1, padding=1)
        self.bottleneck2 = Bottleneck(
            c + 3 * growth_rate, (c + 3 * growth_rate) // 4)

        # Fifth block with mixed convolution types
        self.mixed_conv = nn.ModuleList([
            nn.Conv2d(c + 3 * growth_rate, c + 4 * growth_rate,
                      kernel_size=3, stride=1, padding=1),
            nn.Conv2d(c + 3 * growth_rate, c + 4 * growth_rate,
                      kernel_size=3, stride=1, padding=2, dilation=2),
            nn.Conv2d(c + 3 * growth_rate, c + 4 * growth_rate,
                      kernel_size=1, stride=1, padding=0)
        ])

        self.final_conv = nn.Conv2d(
            c + 4 * growth_rate, c + 4 * growth_rate, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = F.relu(self.initial_conv(x))

        # First bottleneck block
        x = self.bottleneck1(x)

        # Second block with depthwise separable convolution
        x = F.relu(self.depthwise_separable(x))

        # Third block with dilated convolution
        x = F.relu(self.dilated_conv(x))

        # Fourth block with standard convolution and skip connection
        skip = F.relu(self.conv_skip(x))
        x = self.bottleneck2(skip)

        # Fifth block with mixed convolution types
        x_mixed = sum(F.relu(conv(x)) for conv in self.mixed_conv)

        # Final convolution
        x = F.relu(self.final_conv(x_mixed))

        return x
