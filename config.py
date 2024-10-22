from dataclasses import dataclass


@dataclass
class _DASConfig:
    sample_rate: int
    time: int
    space: int
    valid_range: range
    type: str

    @property
    def shape(self):
        return (self.sample_rate * self.time, len(self.valid_range))


DAS_CONFIG = _DASConfig(
    sample_rate=5000,
    time=1,
    space=200,
    valid_range=range(150, 192),
    type="<i2",
)


@dataclass
class _DataConfig:
    downsample: int
    time: int
    space: int
    type: str
    label_map: dict

    @property
    def shape(self):
        return (self.time * DAS_CONFIG.sample_rate // self.downsample, self.space)


DATA_CONFIG = _DataConfig(
    downsample=5,
    time=10,
    space=20,
    type="<f4",
    label_map={"搭梯子": "敲击", "攀爬-弱信号": "未知"},
)
