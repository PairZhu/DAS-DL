# 脚本用于多次训练模型

import os

# 三次训练
train_args = [
    "large_eca_down --epochs 100",
    *(["--epochs 100"] * 4),
]

for arg in train_args:
    try:
        os.system(f"python classify.py {arg}")
    except Exception as e:
        print(e)
        continue
