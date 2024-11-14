import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type

from dwt_module import DWTDownsample
from stft_module import STFTDownsample
from attention import ECABlock, SEBlock


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


class InvertedResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride: int | tuple[int, int] | list[int] = 1,
        ratio: int = 6,
        activation: Type[nn.Module] = nn.ReLU6,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * ratio,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * ratio),
            activation(inplace=True),
            nn.Conv2d(
                out_channels * ratio,
                out_channels * ratio,
                kernel_size=3,
                stride=stride,  # type: ignore
                padding=1,
                groups=out_channels * ratio,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * ratio),
            activation(inplace=True),
            nn.Conv2d(
                out_channels * ratio,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
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
        out = self.conv(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


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


class LargeConvDownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=2, kernel_size=64):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=downsample,
            padding=kernel_size // 2,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out, inplace=True)
        return out


class ClassifyNet(nn.Module):
    def __init__(self):
        super().__init__()

    def feature(self, x):
        raise NotImplementedError

    def classify(self, x):
        raise NotImplementedError

    def cam(self, x):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class DASNet(ClassifyNet):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.downsample = nn.Sequential(
            # bs x 1 x 10000 x 20
            #
            # version 1
            # InceptionDownsampleBlock(in_channels, 64, 0.2, 4),
            # # bs x 64 x 2500 x 20
            # InceptionDownsampleBlock(64, 64, 0.2, 5),
            # # bs x 64 x 500 x 20
            # InceptionDownsampleBlock(64, 64, 0.2, 4),
            # # bs x 64 x 125 x 20
            # InceptionDownsampleBlock(64, 64, 0.2, 5),
            #
            # version 2
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
            #
            # version 3
            # DWTDownsample(in_channels, 2, dim=2),
            # # bs x 2 x 5000 x 20
            # DWTDownsample(2, 4, dim=2),
            # # bs x 4 x 2500 x 20
            # DWTDownsample(4, 8, dim=2),
            # # bs x 8 x 1250 x 20
            # DWTDownsample(8, 16, dim=2),
            # # bs x 16 x 625 x 20
            # DWTDownsample(16, 32, dim=2),
            # # bs x 32 x 313 x 20
            # DWTDownsample(32, 64, dim=2),
            # # bs x 64 x 157 x 20
            # DWTDownsample(64, 64, dim=2),
            # # bs x 64 x 79 x 20
            # DWTDownsample(64, 64, dim=2),
            # # bs x 64 x 40 x 20
            # DWTDownsample(64, 64, dim=2),
            #
            # faster version 3
            # DWTDownsample(in_channels, 4, dim=2, J=2),
            # # bs x 4 x 2500 x 20
            # DWTDownsample(4, 16, dim=2, J=2),
            # # bs x 16 x 625 x 20
            # DWTDownsample(16, 64, dim=2, J=2),
            # # bs x 64 x 157 x 20
            # DWTDownsample(64, 64, dim=2, J=2),
            # # bs x 64 x 40 x 20
            # DWTDownsample(64, 64, dim=2, J=1),
            #
            # version 4
            # STFTDownsample(in_channels, 4, 4, dim=2),
            # # bs x 4 x 2500 x 20
            # STFTDownsample(4, 16, 4, dim=2),
            # # bs x 16 x 625 x 20
            # STFTDownsample(16, 64, 4, dim=2),
            # # bs x 64 x 157 x 20
            # STFTDownsample(64, 64, 4, dim=2),
            # # bs x 64 x 40 x 20
            # STFTDownsample(64, 64, 2, dim=2),
            #
            # faster version 4
            # STFTDownsample(in_channels, 16, 8, dim=2),
            # # bs x 16 x 1250 x 20
            # STFTDownsample(16, 16, 8, dim=2),
            # # bs x 16 x 156 x 20
            # STFTDownsample(16, 64, 8, dim=2),
            #
            # fasteset version 4
            STFTDownsample(
                in_channels,
                64,
                500,
                dim=2,
                window_size=1024,
                keep_time_data=False,
            ),
            #
            # version 5
            # LargeConvDownsampleBlock(in_channels, 2, 2),
            # # bs x 2 x 5000 x 20
            # LargeConvDownsampleBlock(2, 4, 2),
            # # bs x 4 x 2500 x 20
            # LargeConvDownsampleBlock(4, 8, 2),
            # # bs x 8 x 1250 x 20
            # LargeConvDownsampleBlock(8, 16, 2),
            # # bs x 16 x 625 x 20
            # LargeConvDownsampleBlock(16, 32, 2),
            # # bs x 32 x 313 x 20
            # LargeConvDownsampleBlock(32, 64, 2),
            # # bs x 64 x 157 x 20
            # LargeConvDownsampleBlock(64, 64, 2),
            # # bs x 64 x 79 x 20
            # LargeConvDownsampleBlock(64, 64, 2),
            # # bs x 64 x 40 x 20
            # LargeConvDownsampleBlock(64, 64, 2),
            #
            # faster version 5
            # LargeConvDownsampleBlock(in_channels, 16, 8),
            # # bs x 16 x 1250 x 20
            # LargeConvDownsampleBlock(16, 16, 8),
            # # bs x 16 x 156 x 20
            # LargeConvDownsampleBlock(16, 64, 8),
            #
            # version mix
            # STFTDownsample(in_channels, 16, 8, dim=2),
            # # bs x 16 x 1250 x 20
            # ResidualBlock(16, 32, stride=(2, 1)),
            # # bs x 32 x 625 x 20
            # ResidualBlock(32, 64, stride=(2, 1)),
            # # bs x 64 x 313 x 20
            # ResidualBlock(64, 64, stride=(2, 1)),
            # # bs x 64 x 157 x 20
            # ResidualBlock(64, 64, stride=(2, 1)),
            # # bs x 64 x 79 x 20
            # ResidualBlock(64, 64, stride=(2, 1)),
            # # bs x 64 x 40 x 20
            # ResidualBlock(64, 64, stride=(2, 1)),
            #
            #
            ECABlock(64),
            # SEBlock(64, reduction=16),
            # bs x 64 x 20 x 20
        )
        self.backbone = nn.Sequential(
            InvertedResidualBlock(64, 64),
            InvertedResidualBlock(64, 64),
            InvertedResidualBlock(64, 64),
        )
        # bs x 32 x 25 x 20
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # bs x 32 x 1 x 1
        self.fc = nn.Linear(64, num_classes)

        # # Spatial transformer localization-network
        # self.localization = nn.Sequential(
        #     nn.Conv2d(64, 8, kernel_size=7),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True),
        #     nn.Conv2d(8, 10, kernel_size=5),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True),
        #     nn.AdaptiveAvgPool2d((3, 3)),
        # )

        # # Regressor for the 3 * 2 affine matrix
        # self.fc_loc = nn.Sequential(
        #     nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
        # )

        # # Initialize the weights/bias with identity transformation
        # self.fc_loc[2].weight.data.zero_()
        # self.fc_loc[2].bias.data.copy_(
        #     torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        # )

        # Spatial transformer network forward function

    # def stn(self, x):
    #     xs = self.localization(x)
    #     xs = xs.view(-1, 10 * 3 * 3)
    #     theta = self.fc_loc(xs)
    #     theta = theta.view(-1, 2, 3)

    #     grid = F.affine_grid(theta, x.size(), align_corners=False)
    #     x = F.grid_sample(x, grid, align_corners=False)

    #     return x

    def feature(self, x):
        out = self.downsample(x)
        assert out.shape[1:] == (64, 20, 20)
        # out = self.stn(out)
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
