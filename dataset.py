import os
import os.path as osp

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import re

import enhance
from enhance import ProbabilityTransform, RandomChoiceTransform, Transform


def get_params_by_filename(filename):
    # (name:%s)(%d)hz.npy
    match = re.search(r"(.*?)(\d+)hz\.npy$", filename)
    if match:
        name = str(match.group(1))
        hz = int(match.group(2))
        return name, hz
    else:
        raise ValueError(f"Filename {filename} does not match the expected pattern.")


class ClassifyDataset(Dataset):
    SHAPE = (1, 500, 8)

    def __init__(self, root, label_list: list[str], transform: Transform | None = None):
        self.root = root
        self.transform = transform
        self.files = [f for f in os.listdir(root) if f.endswith(".npy")]
        self.labels: list[int] = []
        for file in self.files:
            label = file.split("_")[1]
            self.labels.append(label_list.index(label))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file = self.files[idx]
        data = np.load(osp.join(self.root, file))
        # 归一化
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-6)
        if self.transform:
            data = self.transform(data)
        # 增加通道维度
        data = data[np.newaxis]

        assert data.shape == self.SHAPE
        data = torch.from_numpy(np.ascontiguousarray(data)).float()
        label = torch.tensor(label, dtype=torch.long)
        return data, label


if __name__ == "__main__":
    WINDOW_SIZE = 2  # seconds
    STEP_SIZE = 0.5  # seconds
    FS = 250
    origin_root = osp.join("data", "origin")
    # DP开头的文件视为验证集
    for file in os.listdir(origin_root):
        data = np.load(osp.join(origin_root, file))  # (T,C)
        # 每个样本裁剪出多个样本
        samples = []
        for start in np.arange(0, data.shape[0] / FS, STEP_SIZE):
            start_idx = int(start * FS)
            end_idx = int((start + WINDOW_SIZE) * FS)
            if end_idx > data.shape[0]:
                break
            sample = data[start_idx:end_idx]
            samples.append(sample)
        if file.startswith("DP"):
            save_root = osp.join("data", "val")
        else:
            save_root = osp.join("data", "train")
        os.makedirs(save_root, exist_ok=True)
        name, hz = get_params_by_filename(file)
        for i, sample in enumerate(samples):
            save_path = osp.join(save_root, f"{name}_{hz}hz_{i}.npy")
            np.save(save_path, sample)
