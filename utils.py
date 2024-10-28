from typing import Sequence
import random
from scipy.signal import stft
import numpy as np
import torch
import torch.nn as nn
import pywt
from torch.autograd import Function
import pytorch_wavelets.dwt.lowlevel as lowlevel


def split_array(arr: Sequence, m: int, allow_empty: bool = False) -> list[Sequence]:
    """
    将数组平均分割成(至多)m份

    :param arr: 要分割的数组
    :param m: 份数
    :param allow_empty: 是否允许空数组，如果不允许，则会删除空数组
    :return: 分割后的数组
    """

    n = len(arr)
    # 每一份的基础大小
    base_size = n // m
    # 剩余未能平均分配的元素数
    remainder = n % m

    result = []
    start = 0

    for i in range(m):
        # 如果还有剩余，就在这个份上多分配一个元素
        size = base_size + (1 if i < remainder else 0)
        # 切割数组
        result.append(arr[start : start + size])
        start += size

    # 如果不允许空数组，则删除空数组
    if not allow_empty:
        result = [x for x in result if len(x) > 0]

    return result


class ReservoirSampler:
    """
    蓄水池抽样算法，用于从一个数据流中随机抽取k个样本，保证每个样本被抽取到的概率相等

    :param k: 抽样的样本数量
    """

    def __init__(self, k: int = 1):
        self._k = k
        self._samples = []
        self._idxs = []
        self._cnt = 0

    def process(self, value):
        if len(self._samples) < self._k:
            self._samples.append(value)
            self._idxs.append(self._cnt)
        else:
            idx = random.randint(0, self._cnt)
            if idx < self._k:
                self._samples[idx] = value
                self._idxs[idx] = self._cnt
        self._cnt += 1
        return self._samples

    @property
    def samples(self):
        return self._samples

    @property
    def idxs(self):
        return self._idxs


def short_time_fourier_transform(
    data: np.ndarray,
    window_size: int,
    down_sample: int = 1,
    axis: int = -1,
    **kwargs,
):
    """
    短时傅里叶变换
    :param data: 输入数据
    :param window_size: 窗口大小
    :param down_sample: 下采样倍数
    :param kwargs: 其他参数
    :param axis: 轴
    :return: 频率，数据
    """
    if down_sample >= window_size:
        raise ValueError("Down sample must less than window size")
    pad_before = (window_size - 1) // 2
    last_point = (len(data) - 1) // down_sample * down_sample
    pad_after = window_size // 2 + last_point - len(data) + 1
    pad_width = [(0, 0)] * data.ndim
    pad_width[axis] = (pad_before, pad_after)
    data = np.pad(data, pad_width)
    overlap = window_size - down_sample
    freqs, _, data = stft(
        data,
        nperseg=window_size,
        noverlap=overlap,
        boundary=None,  # type: ignore
        axis=axis,
        **kwargs,
    )
    return freqs, data


class AFB1DFor2D(Function):
    """Does a single level 1d wavelet decomposition of an input.

    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.

    Inputs:
        x (torch.Tensor): Input to decompose
        h0: lowpass
        h1: highpass
        mode (int): use mode_to_int to get the int code here

    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.

    Returns:
        x0: Tensor of shape (N, C, L') - lowpass
        x1: Tensor of shape (N, C, L') - highpass
    """

    @staticmethod
    def forward(ctx, x, h0, h1, mode, dim=3):
        mode = lowlevel.int_to_mode(mode)
        # Save for backwards
        ctx.save_for_backward(h0, h1)
        ctx.shape = x.shape[-2:]
        ctx.mode = mode
        ctx.dim = dim

        lohi = lowlevel.afb1d(x, h0, h1, mode=mode, dim=dim)
        s = lohi.shape
        lohi = lohi.reshape(s[0], -1, 2, s[-2], s[-1])
        lo, hi = torch.unbind(lohi, dim=2)
        return lo.contiguous(), hi.contiguous()

    @staticmethod
    def backward(ctx, dx0, dx1):
        dx = None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            h0, h1 = ctx.saved_tensors
            dim = ctx.dim

            dx = lowlevel.sfb1d(dx0, dx1, h0, h1, mode=mode, dim=dim)

            # Check for odd input
            dx = dx[..., : ctx.shape[-2], : ctx.shape[-1]]  # type: ignore

        return dx, None, None, None, None, None


class DWT1DFor2DForward(nn.Module):
    """Performs a 1d DWT Forward decomposition of an 2d image

    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays (h0, h1)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
    """

    def __init__(self, wave="db1", mode="zero", dim=3):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)  # type: ignore
        if isinstance(wave, pywt.Wavelet):  # type: ignore
            h0, h1 = wave.dec_lo, wave.dec_hi
        elif len(wave) == 2:
            h0, h1 = wave[0], wave[1]

        # Prepare the filters
        filts = lowlevel.prep_filt_afb1d(h0, h1)
        self.register_buffer("h0", filts[0])
        self.register_buffer("h1", filts[1])
        self.mode = mode
        self.dim = dim

    def forward(self, x):
        """Forward pass of the DWT.

        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of length J with the first entry
                being the finest scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        mode = lowlevel.mode_to_int(self.mode)

        yl, yh = AFB1DFor2D.apply(x, self.h0, self.h1, mode, self.dim)  # type: ignore

        return yl, yh
