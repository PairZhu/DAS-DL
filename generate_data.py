from dataclasses import dataclass
import json
import os
import os.path as osp
import numpy as np
import re
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from utils import split_array
from config import DAS_CONFIG, DATA_CONFIG


@dataclass
class Label:
    name: str
    begin: float
    end: float
    top: float
    bottom: float

    def __repr__(self):
        return f"Label({self.name}, {datetime.fromtimestamp(self.begin/1000)}, {datetime.fromtimestamp(self.end/1000)}, {self.top}, {self.bottom})"


@dataclass
class DataFile:
    filename: str
    timestamp: int


class DataGenerator:
    def __init__(
        self,
        root: str,
        data_files: list[DataFile],
        labels: list[Label],
        n_workers: int = 0,
    ):
        self.root = root
        self.data: np.ndarray | None = None
        self.n_workers = n_workers
        # 筛选标签
        file_begin = data_files[0].timestamp
        file_end = data_files[-1].timestamp + DAS_CONFIG.time * 1000
        self.labels: list[Label] = []
        for label in labels:
            if label.begin <= file_begin:
                assert label.end < file_begin, f"标签{label}位于数据段交界处"
                continue
            if label.end >= file_end:
                assert label.begin > file_end, f"标签{label}位于数据段交界处"
                continue
            # 标签名中不应有"_"
            assert "_" not in label.name, f"标签{label}的标签名有下划线"
            # 标签top和bottom应在有效范围内
            assert (
                label.top >= DAS_CONFIG.valid_range.start
                and label.bottom < DAS_CONFIG.valid_range.stop
            ), f"标签{label}的top和bottom不在有效范围内"
            self.labels.append(label)
        # 去除附近无标签的文件
        self.data_files: list[DataFile] = []
        if len(self.labels) == 0:
            return
        label_begin = (
            min([label.begin for label in self.labels]) - DATA_CONFIG.time * 1000
        )
        label_end = max([label.end for label in self.labels]) + DATA_CONFIG.time * 1000
        for file in data_files:
            if file.timestamp + DAS_CONFIG.time * 1000 <= label_begin:
                continue
            if file.timestamp >= label_end:
                continue
            self.data_files.append(file)

    def _read_file(self, file: DataFile):
        file_path = osp.join(self.root, file.filename)
        file_data = np.fromfile(file_path, dtype=DAS_CONFIG.type)
        file_data = file_data.reshape(-1, DAS_CONFIG.space)[:, DAS_CONFIG.valid_range]
        assert file_data.shape == DAS_CONFIG.shape, f"{file.filename}数据形状不符合要求"
        # 释放大数组内存
        return file_data.copy()

    # 读取多个文件
    def _read_files(self, files: list[DataFile]):
        data = []
        for file in files:
            file_data = self._read_file(file)
            data.append(file_data)
        return (
            np.concatenate(data, axis=0)
            if len(data) > 0
            else np.zeros((0, len(DAS_CONFIG.valid_range)))
        )

    def read_data(self):
        if len(self.data_files) == 0:
            self.data = np.zeros((0, len(DAS_CONFIG.valid_range)))
            return
        begin = self.data_files[0].timestamp
        end = self.data_files[-1].timestamp + DAS_CONFIG.time * 1000
        cache_file = osp.join(
            self.root,
            ".cache",
            f"{begin}_{end}_{DAS_CONFIG.valid_range.start}_{DAS_CONFIG.valid_range.stop}.npy",
        )
        if osp.exists(cache_file):
            self.data = np.load(cache_file)
            return
        data: list[np.ndarray] = []
        if self.n_workers <= 0:
            for file_data in map(self._read_file, self.data_files):
                data.append(file_data)
        else:
            # 拆分文件为n_workers份
            task_files = split_array(self.data_files, self.n_workers)
            with ProcessPoolExecutor(self.n_workers) as executor:
                for file_data in executor.map(self._read_files, task_files):
                    data.append(file_data)

        self.data = np.concatenate(data, axis=0)
        os.makedirs(osp.join(self.root, ".cache"), exist_ok=True)
        np.save(cache_file, self.data)

    def release_data(self):
        self.data = None

    def __enter__(self):
        self.read_data()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release_data()

    def _get_intersect_labels(
        self,
        beign: float,
        end: float,
        top: float,
        bottom: float,
        labels: list[Label] | None = None,
    ):
        if labels is None:
            labels = self.labels
        assert labels is not None
        filtered_labels: list[Label] = []
        for label in labels:
            if (
                label.begin < end
                and label.end > beign
                and label.top < bottom
                and label.bottom > top
            ):
                filtered_labels.append(label)
        return filtered_labels

    def _x_to_time(self, x: float):
        assert len(self.data_files) > 0
        return self.data_files[0].timestamp + x / DAS_CONFIG.sample_rate * 1000

    def _time_to_x(self, time: float):
        assert len(self.data_files) > 0
        return (time - self.data_files[0].timestamp) * DAS_CONFIG.sample_rate / 1000

    # 给定一个矩形区域，将label转换为在矩形区域内的坐标，并对超出部分进行裁剪
    def convert_label(
        self, label: Label, left: float, right: float, top: float, bottom: float
    ):
        label_left = self._time_to_x(label.begin)
        label_right = self._time_to_x(label.end)
        x = max(0, label_left - left)
        y = max(0, label.top - top)
        w = min(right, label_right) - max(left, label_left)
        h = min(bottom, label.bottom) - max(top, label.top)
        return x, y, w, h

    def generate_data(
        self,
        left: int,
        top: int,
        min_secs: float = 0.5,
        min_height: int = 2,
        width_percent: float = 0.3,
        height_percent: float = 0.3,
    ) -> tuple[list[Label], list[Label], np.ndarray | None]:
        if self.data is None:
            raise ValueError("未读取数据")
        # 判断范围是否在有效范围内
        right = left + DATA_CONFIG.time * DAS_CONFIG.sample_rate
        bottom = top + DATA_CONFIG.space
        begin = self._x_to_time(left)
        end = self._x_to_time(right)
        if (
            left < 0
            or right > self.data.shape[0]
            or top < DAS_CONFIG.valid_range.start
            or bottom > DAS_CONFIG.valid_range.stop
        ):
            return [], [], None

        # 时间范围内必须有标签
        labels = self._get_intersect_labels(
            begin, end, DAS_CONFIG.valid_range.start, DAS_CONFIG.valid_range.stop
        )

        if len(labels) == 0:
            return [], [], None

        # 筛选出实际相交的区域
        labels = self._get_intersect_labels(begin, end, top, bottom, labels)

        # 标签重命名
        for label in labels:
            for key, value in DATA_CONFIG.label_map.items():
                try:
                    if re.match(key, label.name):
                        label.name = value
                        break
                except re.error:
                    pass
            else:
                label.name = label.name.split("-")[0]

        for label in labels:
            # 实际区域不应包含'未知'标签
            if label.name == "未知":
                return [], [], None

        # 判断是否有'背景'标签或'列车'标签
        has_background = any(
            [label.name == "背景" or label.name == "列车" for label in labels]
        )
        # 删除'背景'标签和'列车'标签
        labels = [
            label for label in labels if label.name != "背景" and label.name != "列车"
        ]

        # 生成数据
        data = self.data[
            left : right : DATA_CONFIG.downsample,
            top - DAS_CONFIG.valid_range.start : bottom - DAS_CONFIG.valid_range.start,
        ].copy()

        # 生成标签
        filtered_labels: list[Label] = []
        for label in labels:
            x, y, w, h = self.convert_label(label, left, right, top, bottom)
            label_width = self._time_to_x(label.end) - self._time_to_x(label.begin)
            label_height = label.bottom - label.top
            # 只保留 时长大于min_secs且高度大于min_height的标签 或
            # 时长大于label_width * width_percent且高度大于label_height * height_percent的标签
            if (w >= DAS_CONFIG.sample_rate * min_secs and h >= min_height) or (
                w >= label_width * width_percent and h >= label_height * height_percent
            ):
                # 适配降采样
                filtered_labels.append(
                    Label(
                        label.name,
                        x // DATA_CONFIG.downsample,
                        (x + w) // DATA_CONFIG.downsample,
                        y,
                        y + h,
                    )
                )
        if filtered_labels == [] and not has_background:
            return [], [], None

        return labels, filtered_labels, data

    def generate_detect_data(
        self, left: int, top: int
    ) -> tuple[list[Label], np.ndarray | None]:
        _, filtered_labels, data = self.generate_data(left, top)
        if data is None:
            return [], data
        begin = int(self._x_to_time(left))
        label_names = set([label.name for label in filtered_labels])
        if len(label_names) == 0:
            label_names.add("背景")
        file_name = f"{begin}_{top}_{'_'.join(label_names)}"
        os.makedirs("detect_data", exist_ok=True)
        np.save(
            osp.join("detect_data", f"{file_name}.npy"),
            data.astype(DATA_CONFIG.type),
        )
        label_file = osp.join("detect_data", f"{file_name}.json")
        with open(label_file, "w", encoding="utf-8") as f:
            json.dump(
                [label.__dict__ for label in filtered_labels], f, ensure_ascii=False
            )
        return filtered_labels, data

    def generate_classify_data(
        self, left: int, top: int
    ) -> tuple[str, np.ndarray | None]:
        origin_labels, filtered_labels, data = self.generate_data(
            left, top, min_secs=2, min_height=3, width_percent=0.5, height_percent=0.5
        )
        if data is None:
            return "", None
        if len(filtered_labels) != len(origin_labels):
            return "", None
        if len(origin_labels) > 1:
            return "", None
        elif len(origin_labels) == 0:
            label_str = "背景"
        else:
            label_str = filtered_labels[0].name

        begin = int(self._x_to_time(left))
        file_name = f"{begin}_{top}_{label_str}"
        os.makedirs("classify_data", exist_ok=True)
        np.save(
            osp.join("classify_data", f"{file_name}.npy"),
            data.astype(DATA_CONFIG.type),
        )
        return label_str, data


def storage_format_to_timestamp(file):
    match = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}.\d{3}", file)
    if not match:
        raise ValueError(f"{file}不符合格式")
    date_str = match.group()
    date = datetime.strptime(date_str, "%Y-%m-%d_%H-%M-%S.%f")
    return int(date.timestamp() * 1000)


def get_data_generators(root, max_depth=10, depth=0):
    if depth == max_depth:
        return []
    generators: list[DataGenerator] = []
    data_files: list[DataFile] = []
    label_file = None
    for file in os.listdir(root):
        # 判断是否有格式为Raw2024-09-11_08-10-00.285.dat的文件
        if file.endswith(".dat") and file.startswith("Raw"):
            try:
                timestamp = storage_format_to_timestamp(file)
            except ValueError:
                continue
            data_files.append(DataFile(file, timestamp))
        # 判断是否有labels.json文件
        elif file == "labels.json":
            label_file = osp.join(root, file)
    # 如果有label文件夹
    if osp.exists(osp.join(root, "label", "labels.json")):
        label_file = osp.join(root, "label", "labels.json")

    if label_file is None or len(data_files) == 0:
        for file in os.listdir(root):
            if osp.isdir(osp.join(root, file)):
                generators.extend(
                    get_data_generators(osp.join(root, file), max_depth, depth + 1)
                )
        return generators

    with open(label_file, "r", encoding="utf-8") as f:
        shapes = json.load(f)["shapes"]
    labels: list[Label] = []
    for shape in shapes:
        if shape["shape_type"] != "rectangle":
            continue
        points = shape["points"]
        begin = float(min([point[0] for point in points]))
        end = float(max([point[0] for point in points]))
        top = float(min([point[1] for point in points]))
        bottom = float(max([point[1] for point in points]))
        assert isinstance(shape["label"], str)
        labels.append(Label(shape["label"], begin, end, top, bottom))
    # 按照起始时间排序
    data_files.sort(key=lambda x: x.timestamp)
    current_files = [data_files[0]]
    for file in data_files[1:]:
        if file.timestamp - current_files[-1].timestamp == DAS_CONFIG.time * 1000:
            current_files.append(file)
        else:
            generators.append(DataGenerator(root, current_files, labels))
            current_files = [file]

    generators.append(DataGenerator(root, current_files, labels))
    return generators


def main():
    root = "label_data"
    generators = get_data_generators(root)
    label_count = {}
    for generator in tqdm(generators):
        generator.n_workers = 8
        with generator:
            assert generator.data is not None
            for left in range(0, generator.data.shape[0], 5000):
                for top in range(
                    DAS_CONFIG.valid_range.start,
                    DAS_CONFIG.valid_range.stop - DATA_CONFIG.space,
                    3,
                ):
                    label_str, data = generator.generate_classify_data(left, top)
                    if data is None:
                        continue
                    if label_str not in label_count:
                        label_count[label_str] = 0
                    label_count[label_str] += 1
        tqdm.write(f"{label_count}")


if __name__ == "__main__":
    main()
