import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils import DWT1DFor2DForward


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
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
        return x * y


class ECA_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA_block, self).__init__()
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
        return x * y.expand_as(x)


class AsymmetricInceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        super().__init__()
        assert len(kernel_sizes) == 3
        assert out_channels % 4 == 0
        out_channels = out_channels // 4
        paddings = []
        for kernel_size in kernel_sizes:
            if isinstance(kernel_size, (list, tuple)):
                paddings.append((kernel_size[0] // 2, kernel_size[1] // 2))
            else:
                paddings.append(kernel_size // 2)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_sizes[0],
            padding=paddings[0],
        )
        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_sizes[1],
            padding=paddings[1],
        )
        self.conv3 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_sizes[2],
            padding=paddings[2],
        )
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_sizes[1], stride=1, padding=paddings[1]),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.pool(x)
        return torch.cat([out1, out2, out3, out4], dim=1)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride: int | tuple[int, int] | list[int] = 1,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,  # type: ignore
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride not in (1, [1, 1], (1, 1)) or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # type: ignore
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out, inplace=True)
        return out


class DownWtOneAxisBlock(nn.Module):
    def __init__(self, in_ch, out_ch, axis=3, J=1):
        super().__init__()
        self.J = J
        self.wt = DWT1DFor2DForward(wave="haar", mode="zero", dim=axis)
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 2**J, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        for _ in range(self.J):
            yl, yh = self.wt(x)
            x = torch.cat([yl, yh], dim=1)
        x = self.conv_bn_relu(x)
        return x


class InceptionDownsampleBlock(nn.Module):
    KERNEL_SIZES = [(1, 1), (3, 3), (7, 7)]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float | None = None,
        stride: int = 2,
    ):
        super().__init__()
        self.inception_block = AsymmetricInceptionBlock(
            in_channels, out_channels, self.KERNEL_SIZES
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(stride, 1))
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        out = self.inception_block(x)
        out = self.pool(out)
        out = F.relu(out, inplace=True)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class ClassifyNet(nn.Module):
    def __init__(self):
        super().__init__()

    def feature(self, _):
        raise NotImplementedError

    def classify(self, _):
        raise NotImplementedError

    def cam(self, _):
        raise NotImplementedError

    def forward(self, _):
        raise NotImplementedError


class DASNet(ClassifyNet):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.downsample = nn.Sequential(
            # version 1
            # # bs x 1 x 10000 x 20
            # DASDownsampleBlock(in_channels, 64, 0.2, 4),
            # # bs x 64 x 2500 x 20
            # DASDownsampleBlock(64, 64, 0.2, 5),
            # # bs x 64 x 500 x 20
            # DASDownsampleBlock(64, 64, 0.2, 4),
            # # bs x 64 x 125 x 20
            # DASDownsampleBlock(64, 64, 0.2, 5),
            # # bs x 64 x 25 x 20
            # version 2
            # bs x 1 x 10000 x 20
            # ResidualBlock(in_channels, 2, stride=(2, 1)),
            # # bs x 2 x 5000 x 20
            # ResidualBlock(2, 4, stride=(2, 1)),
            # # bs x 4 x 2500 x 20
            # ResidualBlock(4, 8, stride=(2, 1)),
            # # bs x 8 x 1250 x 20
            # ResidualBlock(8, 16, stride=(2, 1)),
            # # bs x 16 x 625 x 20
            # ResidualBlock(16, 32, stride=(2, 1)),
            # # bs x 32 x 312 x 20
            # ResidualBlock(32, 64, stride=(2, 1)),
            # # bs x 64 x 156 x 20
            # ResidualBlock(64, 64, stride=(2, 1)),
            # # bs x 64 x 78 x 20
            # ResidualBlock(64, 64, stride=(2, 1)),
            # # bs x 64 x 39 x 20
            # ResidualBlock(64, 64, stride=(2, 1)),
            # bs x 64 x 19 x 20
            # version 3
            # # bs x 1 x 10000 x 20
            # DownWtOneAxisBlock(in_channels, 2, axis=2),
            # # bs x 2 x 5000 x 20
            # DownWtOneAxisBlock(2, 4, axis=2),
            # # bs x 4 x 2500 x 20
            # DownWtOneAxisBlock(4, 8, axis=2),
            # # bs x 8 x 1250 x 20
            # DownWtOneAxisBlock(8, 16, axis=2),
            # # bs x 16 x 625 x 20
            # DownWtOneAxisBlock(16, 32, axis=2),
            # # bs x 32 x 313 x 20
            # DownWtOneAxisBlock(32, 64, axis=2),
            # # bs x 64 x 157 x 20
            # DownWtOneAxisBlock(64, 64, axis=2),
            # # bs x 64 x 79 x 20
            # DownWtOneAxisBlock(64, 64, axis=2),
            # # bs x 64 x 40 x 20
            # DownWtOneAxisBlock(64, 64, axis=2),
            # # bs x 64 x 20 x 20
            # faster version 3
            # bs x 1 x 10000 x 20
            DownWtOneAxisBlock(in_channels, 4, axis=2, J=2),
            # bs x 4 x 2500 x 20
            DownWtOneAxisBlock(4, 16, axis=2, J=2),
            # bs x 16 x 625 x 20
            DownWtOneAxisBlock(16, 64, axis=2, J=2),
            # bs x 64 x 157 x 20
            DownWtOneAxisBlock(64, 64, axis=2, J=2),
            # bs x 64 x 40 x 20
            DownWtOneAxisBlock(64, 64, axis=2, J=1),
            # bs x 64 x 20 x 20
        )
        self.backbone = nn.Sequential(
            ResidualBlock(64, 256),
            ResidualBlock(256, 512),
            ResidualBlock(512, 256),
        )
        # bs x 32 x 25 x 20
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # bs x 32 x 1 x 1
        self.fc = nn.Linear(256, num_classes)

    def feature(self, x):
        out = self.downsample(x)
        out = self.backbone(out)
        return out

    def classify(self, x):
        out = self.avgpool(x)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def cam(self, features):
        feat_size = features.size()
        assert len(feat_size) == 3
        cams: list[torch.Tensor] = []
        features = features.view(features.size(0), -1)
        # c, h, w -> c, h * w
        for i in range(self.fc.out_features):
            weight = self.fc.weight[i]
            # n, c -> c
            cam = weight @ features
            # h * w
            cam = cam.view(feat_size[1:])
            # h, w
            cams.append(cam)
        return cams

    def forward(self, x):
        out = self.feature(x)
        out = self.classify(out)
        return out
