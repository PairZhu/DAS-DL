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
    # 将频率维度合并到通道维度
    spec = spec.permute(0, 1, 3, 2, 4).reshape(
        s[0], s[1] * spec.shape[3], s[2], spec.shape[4]
    )
    if dim != 3:
        spec = spec.transpose(dim, 3)
    return spec


def position_encoding(max_sequence_length, d_model, base=10000):
    pe = torch.zeros(
        max_sequence_length, d_model, dtype=torch.float
    )  # size(max_sequence_length, d_model)
    exp_1 = torch.arange(
        d_model // 2, dtype=torch.float
    )  # 初始化一半维度，sin位置编码的维度被分为了两部分
    exp_value = exp_1 / (d_model / 2)

    alpha = 1 / (base**exp_value)  # size(dmodel/2)
    out = (
        torch.arange(max_sequence_length, dtype=torch.float)[:, None] @ alpha[None, :]
    )  # size(max_sequence_length, d_model/2)
    embedding_sin = torch.sin(out)
    embedding_cos = torch.cos(out)

    pe[:, 0::2] = embedding_sin  # 奇数位置设置为sin
    pe[:, 1::2] = embedding_cos  # 偶数位置设置为cos
    return pe


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
        channel_encoding=True,
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
        self.register_buffer("window", window)
        conv_in_channel = (window_size // 2 + 1) * in_channel
        if keep_time_data:
            conv_in_channel += in_channel
        self.conv = nn.Sequential(
            nn.Conv2d(conv_in_channel, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.channel_encoding = channel_encoding

    def forward(self, x):
        y = stft(
            x,
            window_size=self.window_size,
            hop_size=self.downsample,
            window=self.window,
            dim=self.dim,
        )
        if self.keep_time_data:
            # 对self.dim维度进行降采样
            time_data = x.index_select(
                self.dim,
                torch.arange(0, x.size(self.dim), self.downsample, device=x.device),
            )
            y = torch.cat([time_data, y], dim=1)
        if self.channel_encoding:
            # 对通道维度进行位置编码
            pe = position_encoding(
                max_sequence_length=y.size(1), d_model=y.size(2) * y.size(3), base=10000
            )
            pe = pe.to(y.device)
            pe = pe.view(1, pe.size(0), y.size(2), y.size(3))
            y = y + pe

        y = self.conv(y)
        return y
