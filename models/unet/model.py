import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import get


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.double_conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, num_classes=19, in_channels=3, base_channels=64):
        super().__init__()

        c0 = base_channels
        c1 = base_channels * 2
        c2 = base_channels * 4
        c3 = base_channels * 8

        self.inc = DoubleConv(in_channels, c0)
        self.down1 = Down(c0, c1)
        self.down2 = Down(c1, c2)
        self.down3 = Down(c2, c3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.bottleneck = DoubleConv(c3, c3 * 2)

        self.up1 = Up(c3 * 2, c3)
        self.up2 = Up(c3, c2)
        self.up3 = Up(c2, c1)
        self.up4 = Up(c1, c0)

        self.out = nn.Conv2d(c0, num_classes, kernel_size=1)

    def forward(self, x):
        e0 = self.inc(x)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.down3(e2)

        b = self.bottleneck(self.pool(e3))

        d1 = self.up1(b, e3)
        d2 = self.up2(d1, e2)
        d3 = self.up3(d2, e1)
        d4 = self.up4(d3, e0)

        return self.out(d4)

    @classmethod
    def from_config(cls, cfg):
        model_cfg = get(cfg, "model", {}) or {}
        return cls(
            num_classes=get(model_cfg, "num_classes", 19),
            in_channels=get(model_cfg, "in_channels", 3),
            base_channels=get(model_cfg, "base_channels", 64),
        )