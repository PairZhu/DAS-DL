import torch
import torch.nn.functional as F
import torch.nn as nn


def stft(x: torch.Tensor, window_size, hop_size, window=None, dim=-1):
    # x: (B, C, H, W)
    device = x.device
    if window is None:
        window = torch.ones(window_size, device=device)
    else:
        window = window.to(device)
    dim = dim % x.dim()
    if dim != 3:
        x = x.transpose(dim, 3)
    s = x.shape
    x = x.reshape(-1, s[3])
    pad_left = window_size // 2
    pad_right = window_size - pad_left - 1
    x = F.pad(x, (pad_left, pad_right))
    spec = torch.stft(
        x,
        n_fft=window_size,
        hop_length=hop_size,
        window=window,
        center=False,
        return_complex=True,
    ).abs()
    spec = spec.view(
        s[0], s[1], s[2], spec.shape[-2], spec.shape[-1]
    )  # (B, C, H, F, W)
    # 将频率维度放在通道维度后面
    spec = spec.permute(0, 1, 3, 2, 4)  # (B, C, F, H, W)
    if dim != 3:
        spec = spec.transpose(dim + 1, 4)
    return spec


class FreqAttention(nn.Module):
    def __init__(self, freq_channel, reduction=4):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(3, freq_channel // reduction, 1, bias=False)
        self.bn = nn.BatchNorm1d(freq_channel // reduction)
        self.fc = nn.Linear(freq_channel // reduction, 1)

        self.register_buffer("freq_weight", torch.arange(freq_channel) / freq_channel)

    def forward(self, x):
        # x: (B, C, F, H, W)
        b, c, f, h, w = x.size()
        x = x.view(b * c, f, h, w)
        # (B*C, F, H, W)
        max_out = self.max_pool(x).view(b * c, 1, f)
        avg_out = self.avg_pool(x).view(b * c, 1, f)
        freq_out = self.freq_weight.view(1, 1, f).expand(b * c, 1, f)
        out = torch.cat([max_out, avg_out, freq_out], dim=1)  # (B*C, 3, F)
        out = self.conv(out)  # (B*C, F//reduction, F)
        out = self.bn(out)
        out = F.relu(out, inplace=True)

        # 交换维度 B*C, F//reduction, F -> B*C, F, F//reduction
        out = out.transpose(1, 2)
        out = self.fc(out)  # (B*C, F, 1)
        mask = torch.sigmoid(out)

        y = x * mask.view(b * c, f, 1, 1)
        y += x
        y = y.view(b, c, f, h, w)
        return y


class STFTDownsample(torch.nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        downsample,
        window_size=64,
        dim=3,
        window=None,
        keep_time_data=True,
        freq_attention=True,
    ):
        super().__init__()
        if downsample >= window_size:
            raise ValueError("Down sample must less than window size")
        self.downsample = downsample
        self.window_size = window_size
        self.dim = dim
        self.keep_time_data = keep_time_data
        if window is None:
            window = torch.ones(window_size)
        stft_channel = in_channel
        self.register_buffer("window", window)
        freq_channel = window_size // 2 + 1
        conv_in_channel = freq_channel * stft_channel
        self.freq_attention = FreqAttention(freq_channel) if freq_attention else None
        if keep_time_data:
            conv_in_channel += stft_channel
        self.conv = nn.Sequential(
            nn.Conv2d(conv_in_channel, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        y = stft(
            x,
            window_size=self.window_size,
            hop_size=self.downsample,
            window=self.window,
            dim=self.dim,
        )
        # (B, C, F, H, W)
        if self.freq_attention:
            y = self.freq_attention(y)

        y = y.reshape(y.size(0), -1, y.size(-2), y.size(-1))
        # (B, C*F, H, W)

        if self.keep_time_data:
            # 对self.dim维度进行降采样
            time_data = x.index_select(
                self.dim,
                torch.arange(0, x.size(self.dim), self.downsample, device=x.device),
            )
            y = torch.cat([time_data, y], dim=1)

        y = self.conv(y)
        return y
