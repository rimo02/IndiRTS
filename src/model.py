import torch
import torch.nn as nn
import torch.nn.functional as F
from .fdm import FDM
from .dlm import DLM1, DLM2


class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.fdm1 = FDM(3, 16)
        self.fdm2 = FDM(16, 32)
        self.fdm3 = FDM(32, 64)
        self.dlm1 = DLM1(16, 12)
        self.dlm2 = DLM2(16, 12)
        self.c1 = nn.Conv2d(in_channels=64, out_channels=64,
                            kernel_size=1, stride=1, padding=0)
        self.c2 = nn.Conv2d(in_channels=64, out_channels=64,
                            kernel_size=1, stride=1, padding=0)
        self.c3 = nn.Conv2d(in_channels=64, out_channels=32,
                            kernel_size=1, stride=1, padding=0)
        self.c4 = nn.Conv2d(in_channels=32, out_channels=16,
                            kernel_size=1, stride=1, padding=0)
        self.c5 = nn.Conv2d(in_channels=16, out_channels=3,
                            kernel_size=1, stride=1, padding=0)
        self.c6 = nn.Conv2d(
            in_channels=3, out_channels=self.num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1, x2 = self.fdm1(x)
        x3, x4 = self.fdm2(x2)
        x5, x6 = self.fdm3(x4)
        x7, x8 = self.dlm1(x6)
        x9, x10 = self.dlm2(x8)

        x11 = F.relu(self.c1(x10))
        x12 = F.interpolate(x11, scale_factor=2, mode='bilinear')
        out5 = x9 + x12

        x13 = F.relu(self.c2(out5))
        x14 = F.interpolate(x13, scale_factor=2, mode='bilinear')
        out4 = x7 + x14

        x15 = F.relu(self.c3(out4))
        x16 = F.interpolate(x15, scale_factor=2, mode='bilinear')
        out3 = x5 + x16

        x17 = F.relu(self.c4(out3))
        x18 = F.interpolate(x17, scale_factor=2, mode='bilinear')
        out2 = x3 + x18

        x19 = F.relu(self.c5(out2))
        x20 = F.interpolate(x19, scale_factor=2, mode='bilinear')
        out1 = x1 + x20

        out = F.softmax(self.c6(out1), dim=1)
        return out


def CreateModel(num_class):
    crearedmodel =  Model(num_classes=num_class)
    return crearedmodel
