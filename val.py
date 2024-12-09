# 脚本用于连续验证模型的性能

import os

# .\classify.py val -m .\runs\2024-11-17_23-49-46_double-eca\checkpoints\best.pth --val .\data\test
all_runs = os.listdir("./runs")[-4:-3]
all_runs = sorted(all_runs)
val_args = [
    f"val -m ./runs/{run}/checkpoints/{model} --val ./data/test -w 0"
    for run in all_runs
    for model in os.listdir(f"./runs/{run}/checkpoints")
    if model.startswith("best")
]

for arg in val_args:
    print(arg)
    try:
        os.system(f"python classify.py {arg}")
    except Exception as e:
        print(e)
        continue
