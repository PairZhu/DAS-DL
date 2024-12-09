# 脚本用于多次训练模型

import os

seeds = [0, 1, 42, 1024]
train_args = [
    f"--lr 1e-6 -b 64 -w 8 -e 200 --seed {seeds[0]}",
    *([f"--lr 1e-6 -b 64 -w 8 -e 200 --seed {seed}" for seed in seeds[1:]]),
]

for arg in train_args:
    try:
        os.system(f"python classify.py {arg}")
    except Exception as e:
        print(e)
        continue
