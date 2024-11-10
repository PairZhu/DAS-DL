import math
import torch.nn as nn
import torch


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y + x


class ECABlock(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super().__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x) + x


class CBAMBlock(nn.Module):
    class ChannelAttention(nn.Module):
        def __init__(self, channel, reduction=16):
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.fc = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(channel // reduction, channel, 1, bias=False),
            )
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = self.fc(self.avg_pool(x))
            max_out = self.fc(self.max_pool(x))
            return self.sigmoid(avg_out + max_out) * x

    class SpatialAttention(nn.Module):
        def __init__(self):
            super().__init__()
            kernel_size = 7
            self.conv = nn.Conv2d(
                2, 1, kernel_size, padding=kernel_size // 2, bias=False
            )
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            x = torch.cat([avg_out, max_out], dim=1)
            x = self.conv(x)
            return self.sigmoid(x) * x

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.ca = self.ChannelAttention(channel, reduction)
        self.sa = self.SpatialAttention()

    def forward(self, x):
        y = self.ca(x)
        y = self.sa(y)
        return y + x
