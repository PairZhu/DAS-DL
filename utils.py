from typing import Sequence
import random
from scipy.signal import stft
import numpy as np
import matplotlib.pyplot as plt


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


def visualize_cam(input_data, cam_tensor_list):
    # CAM可视化
    nimgs = len(cam_tensor_list) + 1
    nrows = int(nimgs**0.5)
    ncols = int(np.ceil(nimgs / nrows))
    fig, axes = plt.subplots(nrows, ncols)
    axes = axes.flatten()
    # 对数变换，增强对比度
    input_data = np.abs(input_data, out=input_data)
    input_data = np.log1p(input_data, out=input_data)
    axes[0].imshow(
        input_data,
        cmap="jet",
        vmin=0,
        vmax=np.log1p(1000),
        aspect="auto",
        interpolation="bilinear",
    )
    axes[0].axis("off")
    # 生成CAM
    vmax = max(cam.max().item() for cam in cam_tensor_list)
    vmin = min(cam.min().item() for cam in cam_tensor_list)
    for class_idx, cam_tensor in enumerate(cam_tensor_list):
        cam = cam_tensor.cpu().numpy()
        axes[class_idx + 1].imshow(
            cam,
            cmap="jet",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="bilinear",
        )
        axes[class_idx + 1].axis("off")
    for i in range(len(cam_tensor_list) + 1, nrows * ncols):
        fig.delaxes(axes[i])
    axes = axes[: len(cam_tensor_list) + 1]
    return fig, axes
