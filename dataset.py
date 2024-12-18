import os
import os.path as osp

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import enhance
from config import DATA_CONFIG
from enhance import ProbabilityTransform, RandomChoiceTransform, Transform


def get_params_by_filename(filename):
    filename = osp.splitext(filename)[0]
    params = filename.split("_")
    timestamp = int(params[0])
    top = int(params[1])
    labels = params[2:]
    # 清除空标签
    labels = [label for label in labels if label]
    return timestamp, top, labels


class ClassifyDataset(Dataset):
    DOWN_SAMPLE = 1
    SHAPE = (1, DATA_CONFIG.shape[0] // DOWN_SAMPLE, DATA_CONFIG.shape[1])

    def __init__(self, root, label_list: list[str], transform: Transform | None = None):
        self.root = root
        self.transform = transform
        self.files = [f for f in os.listdir(root) if f.endswith(".npy")]
        self.labels: list[int] = []
        for file in self.files:
            _, _, labels = get_params_by_filename(file)
            assert len(labels) == 1
            self.labels.append(label_list.index(labels[0]))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file = self.files[idx]
        data = np.load(osp.join(self.root, file))
        if self.transform:
            data = self.transform(data)
        # 下采样
        data = data[:: self.DOWN_SAMPLE]
        # 增加通道维度
        data = data[np.newaxis]

        assert data.shape == self.SHAPE
        data = torch.from_numpy(np.ascontiguousarray(data)).float()
        label = torch.tensor(label, dtype=torch.long)
        return data, label


# 检查两个文件的时空位置是否有重合
def check_overlap(file1, file2):
    begin1, top1, _ = get_params_by_filename(file1)
    begin2, top2, _ = get_params_by_filename(file2)
    end1 = begin1 + DATA_CONFIG.time * 1000
    end2 = begin2 + DATA_CONFIG.time * 1000
    bottom1 = top1 + DATA_CONFIG.space
    bottom2 = top2 + DATA_CONFIG.space
    x_overlap = begin1 < end2 and end1 > begin2
    y_overlap = top1 < bottom2 and bottom1 > top2
    return x_overlap, y_overlap


def split_files(files, ratio=0.8):
    label_files = {}
    if isinstance(files, dict):
        files = [file for item in files.values() for file in item]
    for file in files:
        timestamp, top, labels = get_params_by_filename(file)
        if len(labels) > 1:
            continue
        else:
            label = labels[0].split("-")[0]
        if label not in label_files:
            label_files[label] = []
        label_files[label].append((timestamp, file))

    # 对每个类别的文件按时间戳排序
    for label in label_files:
        label_files[label].sort(key=lambda x: x[0])

    train_files: dict[str, list[str]] = {}
    test_files: dict[str, list[str]] = {}

    for label in label_files:
        files = label_files[label]
        split_idx = int(len(files) * ratio)
        train_files[label] = [file for _, file in files[:split_idx]]
        test_files[label] = [file for _, file in files[split_idx:]]
        filtered_files = []
        for test_file in test_files[label]:
            overlaped = False
            for train_file in reversed(train_files[label]):
                x_overlap, y_overlap = check_overlap(train_file, test_file)
                # 如果x轴不重合说明之后的文件也不会重合
                if not x_overlap:
                    break
                # 如果y轴不重合则继续检查下一个文件
                if not y_overlap:
                    continue
                # 如果有重合则后面不用再检查
                else:
                    overlaped = True
                    break
            if not overlaped:
                filtered_files.append(test_file)
        test_files[label] = filtered_files

    return train_files, test_files


def balance_files(root, max_len=None, min_enhance=0.2):
    files = [f for f in os.listdir(root) if f.endswith(".npy")]
    label_files: dict[str, list[str]] = {}
    for file in files:
        _, _, labels = get_params_by_filename(file)
        assert len(labels) == 1
        label = labels[0]
        if label not in label_files:
            label_files[label] = []
        label_files[label].append(file)

    # 打乱文件
    for label in label_files:
        np.random.shuffle(label_files[label])

    # 削减过多的文件
    if max_len:
        # 根据最小增强比例计算保留的文件数量
        remain_len = int(max_len // (1 + min_enhance))
        for label in label_files:
            remove_files = label_files[label][remain_len:]
            label_files[label] = label_files[label][:remain_len]
            for file in tqdm(remove_files, desc=f"Remove {label} Files"):
                os.remove(osp.join(root, file))
    else:
        max_len = max(len(files) for files in label_files.values())

    transform = ProbabilityTransform(
        [
            RandomChoiceTransform(
                [
                    enhance.GuassianNoise(std=lambda: np.random.uniform(0.1, 0.5)),
                    enhance.RandomNoise(low=-1, high=1),
                ]
            ),
            enhance.Flip(axis=0),
            enhance.Flip(axis=1),
            enhance.ScaleValue(scale=lambda: np.random.uniform(0.9, 1.1)),
            enhance.OffsetValue(offset=lambda: np.random.uniform(-1, 1)),
        ],
        p=[1, 0.5, 0.5, 1, 1],
    )

    # 数据增强来平衡数据
    for label in label_files:
        files = label_files[label]
        for i in tqdm(range(max_len - len(files)), desc=f"Enhance {label}"):
            file = np.random.choice(files)
            data = np.load(osp.join(root, file))
            data = transform(data)
            timestamp, _, _ = get_params_by_filename(file)
            new_file = f"{timestamp}_{-i}_{label}.npy"
            np.save(osp.join(root, new_file), data)


def remove_remark(file):
    base, ext = osp.splitext(file)
    base = base.rsplit("-", 1)[0]
    return base + ext


if __name__ == "__main__":
    source_files = [f for f in os.listdir("classify_data") if f.endswith(".npy")]
    train_files, other_files = split_files(source_files, 0.6)
    val_files, test_files = split_files(other_files, 0.5)

    train_path = osp.join("data", "train")
    val_path = osp.join("data", "val")
    test_path = osp.join("data", "test")

    train_file_count = sum(len(files) for files in train_files.values())
    val_file_count = sum(len(files) for files in val_files.values())
    test_file_count = sum(len(files) for files in test_files.values())
    print(f"Train: {train_file_count}, Val: {val_file_count}, Test: {test_file_count}")
    print(
        "Train: {}".format({label: len(files) for label, files in train_files.items()})
    )
    print("Val: {}".format({label: len(files) for label, files in val_files.items()}))
    print("Test: {}".format({label: len(files) for label, files in test_files.items()}))

    confirm = input("Continue? (y/n): ")
    if confirm != "y":
        exit()

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    with tqdm(total=train_file_count, desc="Link Train Files") as pbar:
        for label, files in train_files.items():
            for file in files:
                new_file = remove_remark(file)
                os.link(osp.join("classify_data", file), osp.join(train_path, new_file))
                pbar.update(1)

    with tqdm(total=val_file_count, desc="Link Val Files") as pbar:
        for label, files in val_files.items():
            for file in files:
                new_file = remove_remark(file)
                os.link(osp.join("classify_data", file), osp.join(val_path, new_file))
                pbar.update(1)

    with tqdm(total=test_file_count, desc="Link Test Files") as pbar:
        for label, files in test_files.items():
            for file in files:
                new_file = remove_remark(file)
                os.link(osp.join("classify_data", file), osp.join(test_path, new_file))
                pbar.update(1)
