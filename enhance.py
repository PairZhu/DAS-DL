from abc import ABC, abstractmethod
import numpy as np
import cv2
from typing import Callable


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
