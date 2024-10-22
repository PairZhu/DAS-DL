import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import natsort
import argparse

# 支持中文
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

parser = argparse.ArgumentParser()
# 若干个路径
parser.add_argument("data_path", type=str)
args = parser.parse_args()
TARGET_PATH = args.data_path
DATA_FILES = natsort.natsorted(os.listdir(TARGET_PATH))

data_index = 0
file_data = None

fig, ax = plt.subplots()
plt.axis("off")


def show_data():
    global ax, data_index, file_data
    # 标题显示当前标签类型和标签文件名
    data_file = DATA_FILES[data_index]
    file_path = osp.join(TARGET_PATH, data_file)
    file_data = np.load(file_path)[::10].T
    data = np.abs(file_data)
    data = np.log1p(np.abs(data))
    ax.set_title(f"{data_index}. {data_file}")
    ax.imshow(data, aspect="auto", cmap="jet", vmin=0, vmax=np.log1p(1000))


def on_key(event):
    global data_index
    middle_step = len(DATA_FILES) // 100
    long_step = len(DATA_FILES) // 10
    if event.key == "left":
        data_index -= 1
    elif event.key == "right":
        data_index += 1
    elif event.key == "up":
        data_index -= long_step
    elif event.key == "down":
        data_index += long_step
    elif event.key == "a":
        data_index -= middle_step
    elif event.key == "d":
        data_index += middle_step
    else:
        return
    data_index = data_index % len(DATA_FILES)
    show_data()
    plt.draw()


show_data()
fig.canvas.mpl_connect("key_press_event", on_key)
# 设置窗口标题
title = osp.basename(osp.normpath(TARGET_PATH))
assert fig.canvas.manager is not None
fig.canvas.manager.set_window_title(title)
plt.show()
