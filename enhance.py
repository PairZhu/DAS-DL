from abc import ABC, abstractmethod
import numpy as np
import cv2
from typing import Callable
from scipy.signal import butter, filtfilt


def guassian_noise(image: np.ndarray, mean: float = 0, std: float = 0.1) -> np.ndarray:
    """
    给图像添加高斯噪声

    :param image: 输入图像
    :param mean: 均值
    :param std: 标准差
    :return: 添加噪声后的图像
    """
    noise = np.random.normal(mean, std, image.shape)
    return image + noise


def random_noise(image: np.ndarray, low: float = 0, high: float = 1) -> np.ndarray:
    """
    给图像添加随机噪声

    :param image: 输入图像
    :param low: 噪声下界
    :param high: 噪声上界
    :return: 添加噪声后的图像
    """
    noise = np.random.uniform(low, high, image.shape)
    return image + noise


def flip(image: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    翻转图像

    :param image: 输入图像
    :param axis: 翻转轴
    :return: 翻转后的图像
    """
    return np.flip(image, axis)


def scale_value(image: np.ndarray, scale: float) -> np.ndarray:
    """
    按比例缩放图像的值

    :param image: 输入图像
    :param scale: 缩放比例
    :return: 缩放后的图像
    """
    return image * scale


def offset_value(image: np.ndarray, offset: float) -> np.ndarray:
    """
    偏移图像的值

    :param image: 输入图像
    :param offset: 偏移量
    :return: 偏移后的图像
    """
    return image + offset


def guassian_blur(
    image: np.ndarray, kernel_size: int = 3, sigma: float = 0
) -> np.ndarray:
    """
    对图像进行高斯模糊

    :param image: 输入图像
    :param kernel_size: 模糊核大小
    :param sigma: 标准差
    :return: 模糊后的图像
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


class Transform(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, data: np.ndarray) -> np.ndarray:
        pass


class GuassianNoise(Transform):
    def __init__(
        self,
        mean: float | Callable[[], float] = 0,
        std: float | Callable[[], float] = 0.1,
    ):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        mean = self.mean() if callable(self.mean) else self.mean
        std = self.std() if callable(self.std) else self.std
        return guassian_noise(data, mean, std)


class RandomNoise(Transform):
    def __init__(
        self,
        low: float | Callable[[], float] = 0,
        high: float | Callable[[], float] = 1,
    ):
        self.low = low
        self.high = high

    def __call__(self, data):
        low = self.low() if callable(self.low) else self.low
        high = self.high() if callable(self.high) else self.high
        return random_noise(data, low, high)


class Flip(Transform):
    def __init__(self, axis: int = 0, p: float = 0.5):
        self.axis = axis

    def __call__(self, data):
        return flip(data, self.axis)


class ScaleValue(Transform):
    def __init__(self, scale: float | Callable[[], float] = 1):
        self.scale = scale

    def __call__(self, data):
        scale = self.scale() if callable(self.scale) else self.scale
        return scale_value(data, scale)


class OffsetValue(Transform):
    def __init__(self, offset: float | Callable[[], float] = 0):
        self.offset = offset

    def __call__(self, data):
        offset = self.offset() if callable(self.offset) else self.offset
        return offset_value(data, offset)


class GuassianBlur(Transform):
    def __init__(
        self,
        kernel_size: int | Callable[[], int] = 3,
        sigma: float | Callable[[], float] = 0,
    ):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, data):
        kernel_size = (
            self.kernel_size() if callable(self.kernel_size) else self.kernel_size
        )
        sigma = self.sigma() if callable(self.sigma) else self.sigma
        return guassian_blur(data, kernel_size, sigma)


class SequentialTransform(Transform):
    def __init__(self, transforms: list[Transform]):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data


class ProbabilityTransform(Transform):
    def __init__(self, transforms: list[Transform], p: float | list[float]):
        self.transforms = transforms
        if not isinstance(p, list):
            p = [float(p)] * len(transforms)
        if len(p) != len(transforms):
            raise ValueError(
                "The length of p should be equal to the length of transform"
            )
        self.p = p

    def __call__(self, data):
        for transform, p in zip(self.transforms, self.p):
            if np.random.rand() < p:
                data = transform(data)
        return data


class RandomChoiceTransform(Transform):
    def __init__(
        self, transforms: list[Transform], n: int = 1, p: list[float] | None = None
    ):
        self.transforms = transforms
        self.n = n
        self.p = p

    def __call__(self, data):
        idxs = np.random.choice(
            range(len(self.transforms)), size=self.n, replace=False, p=self.p
        )
        idxs.sort()
        for idx in idxs:
            data = self.transforms[int(idx)](data)
        return data


class EEGDryElectrodeAugment(Transform):
    """
    干电极 EEG 仿真增强（相对尺度）
    data shape: (T, C)
    """

    def __init__(
        self,
        fs: int = 250,
        p_white: float = 1.0,
        p_drift: float = 0.7,
        p_powerline: float = 0.5,
        p_motion: float = 0.3,
    ):
        self.fs = fs
        self.p_white = p_white
        self.p_drift = p_drift
        self.p_powerline = p_powerline
        self.p_motion = p_motion

    def __call__(self, data: np.ndarray) -> np.ndarray:
        data = data.copy()
        T, C = data.shape

        # ===== 1. per-channel 鲁棒尺度（MAD）=====
        # s_c shape: (C,)
        s_c = np.median(np.abs(data - np.median(data, axis=0)), axis=0)
        s_c[s_c == 0] = np.mean(s_c[s_c > 0])  # 防止全零通道

        # ===== 2. 白噪声（热噪声等效）=====
        if np.random.rand() < self.p_white:
            alpha = np.random.uniform(0.05, 0.15)
            white = np.random.randn(T, C) * (alpha * s_c)
            data += white

        # ===== 3. 低频漂移 / 极化噪声（1/f 近似）=====
        if np.random.rand() < self.p_drift:
            beta = np.random.uniform(0.1, 0.4)
            drift = self._low_freq_noise(T, C, cutoff=0.5)
            data += beta * s_c * drift

            # 共模漂移（更真实）
            if np.random.rand() < 0.5:
                cm_beta = np.random.uniform(0.05, 0.2)
                cm = self._low_freq_noise(T, 1, cutoff=0.3)
                data += cm_beta * np.mean(s_c) * cm

        # ===== 4. 工频干扰（50Hz）=====
        if np.random.rand() < self.p_powerline:
            gamma = np.random.uniform(0.05, 0.2)
            t = np.arange(T) / self.fs
            phase = np.random.uniform(0, 2 * np.pi, size=C)
            amp = gamma * s_c * np.random.uniform(0.8, 1.2, size=C)
            powerline = np.sin(2 * np.pi * 50 * t[:, None] + phase) * amp
            data += powerline

        # ===== 5. 运动伪迹（瞬态）=====
        if np.random.rand() < self.p_motion:
            n_events = np.random.randint(1, 4)
            for _ in range(n_events):
                ch = np.random.randint(0, C)
                t0 = np.random.randint(0, T - 20)
                duration = np.random.randint(20, 80)
                delta = np.random.uniform(0.5, 2.0) * s_c[ch]

                envelope = np.exp(-np.linspace(0, 3, duration))
                data[t0 : t0 + duration, ch] += delta * envelope[: T - t0]

        return data

    def _low_freq_noise(self, T, C, cutoff=0.5):
        """生成低频噪声（白噪声 + 低通滤波）"""
        noise = np.random.randn(T, C)
        b, a = butter(2, cutoff / (self.fs / 2), btype="low")
        return filtfilt(b, a, noise, axis=0)