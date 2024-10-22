import torch
import torch.nn as nn
import torch.nn.functional as F


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


class DASDownsampleBlock(nn.Module):
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


class DASDownsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.inception_blocks = nn.Sequential(
            # bs x 64 x 100 x 20
            DASDownsampleBlock(in_channels, 64, 0.2),
            # bs x 64 x 50 x 20
            DASDownsampleBlock(64, 64, 0.5),
            # bs x 64 x 25 x 20
        )

    def forward(self, x):
        out = self.inception_blocks(x)
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
        self.downsample = DASDownsample(in_channels)
        # bs x 64 x 25 x 20
        self.backbone = nn.Sequential(
            ResidualBlock(64, 256),
            ResidualBlock(256, 64),
            ResidualBlock(64, 32),
        )
        # bs x 32 x 25 x 20
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # bs x 32 x 1 x 1
        self.fc = nn.Linear(32, num_classes)

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
