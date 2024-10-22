from typing import Sequence
import random
from scipy.signal import stft
import numpy as np
import torch


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
