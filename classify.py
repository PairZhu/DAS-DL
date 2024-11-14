import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard.writer import SummaryWriter
import os
import os.path as osp
import shutil
import numpy as np
import random
import argparse
from typing import Callable
from functools import partial
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import thop
import copy
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import importlib

from model import DASNet, ClassifyNet
from dataset import ClassifyDataset
import enhance
from enhance import RandomChoiceTransform, ProbabilityTransform
from utils import ReservoirSampler, visualize_cam

LABEL_LIST = ["背景", "敲击", "攀爬", "连续振动"]
# 兼容中文
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
# 启用cudnn加速
torch.backends.cudnn.benchmark = True


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(
    data_loader: DataLoader,
    model: ClassifyNet,  # type: ignore
    optimizer: optim.Optimizer,
    epochs: int | range,
    callback: Callable | None = None,
    device: torch.device | str = "cuda",
):
    model = model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    if isinstance(epochs, int):
        epochs = range(epochs)
    for epoch in epochs:
        total = 0
        correct = 0
        total_loss = 0
        for inputs, labels in tqdm(data_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(
            f"Epoch {epoch}/{epochs[-1]}, Loss: {total_loss/len(data_loader):.4f}, Accuracy: {correct / total:.4f}"
        )
        writer.add_scalar("Loss/train", total_loss / len(data_loader), epoch)
        writer.add_scalar("Accuracy/train", correct / total, epoch)
        if callback:
            callback(epoch)
    writer.flush()


@torch.no_grad()
def validate(
    data_loader: DataLoader,
    model: ClassifyNet,  # type: ignore
    device: torch.device | str = "cuda",
    epoch: int = 0,
):
    model = model.to(device)
    model.eval()
    all_preds = []
    ground_truths = []
    # 各类随机采样一个数据用于CAM可视化激活图
    cam_samplers = [ReservoirSampler(1) for _ in LABEL_LIST]
    for inputs, labels in tqdm(data_loader, desc="Validating"):
        features = model.feature(inputs.to(device))
        outputs = model.classify(features)
        _, predicted = torch.max(outputs.data, 1)
        for i in range(len(labels)):
            label = labels[i].item()
            assert isinstance(label, int)
            cams = model.cam(features[i])
            cam_samplers[label].process((inputs[i], cams, predicted[i]))
        all_preds.extend(predicted.cpu().numpy())
        ground_truths.extend(labels.numpy())

    # 分类报告
    print(
        classification_report(
            ground_truths,
            all_preds,
            target_names=LABEL_LIST,
            zero_division=0,
            digits=4,
        )
    )
    f1_macro = f1_score(ground_truths, all_preds, average="macro")
    f1_weighted = f1_score(ground_truths, all_preds, average="weighted")
    writer.add_scalar("F1/Macro", f1_macro, epoch)
    writer.add_scalar("F1/Weighted", f1_weighted, epoch)

    # 混淆矩阵
    cm = confusion_matrix(ground_truths, all_preds, normalize="true")
    fig = sns.heatmap(
        cm,
        annot=True,
        xticklabels=list(LABEL_LIST),
        yticklabels=list(LABEL_LIST),
        cmap="Blues",
    ).get_figure()
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.close(fig)
    assert fig is not None
    writer.add_figure("Confusion Matrix", fig, global_step=epoch)

    # CAM可视化
    for truth, sampler in enumerate(cam_samplers):
        input_tensor, cam_tensor_list, predicted_tensor = sampler.samples[0]
        input_data = input_tensor.mean(dim=0).cpu().numpy()
        fig, axes = visualize_cam(input_data, cam_tensor_list)
        input_idx = sampler.idxs[0]
        axes[0].set_title(f"Input {input_idx}")
        predicted = predicted_tensor.item()
        for i, ax in enumerate(axes[1:]):
            if i == predicted:
                ax.set_title(f"{LABEL_LIST[i]}(Predicted)")
            else:
                ax.set_title(f"{LABEL_LIST[i]}")
        plt.tight_layout()
        plt.close(fig)
        writer.add_figure(f"CAM/{LABEL_LIST[truth]}", fig, global_step=epoch)

    writer.flush()
    return f1_weighted


def main():
    model = DASNet(1, len(LABEL_LIST))

    if args.model:
        checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
        # 可能是权重，也可能是整个模型
        try:
            model.load_state_dict(checkpoint["model"].state_dict())
        except RuntimeError:
            print("Model Structure Mismatch, Try Load Whole Model")
            model = checkpoint["model"]
        except AttributeError:
            model.load_state_dict(checkpoint["model"])
        begin_epoch = checkpoint.get("epoch", -1) + 1
    else:
        begin_epoch = 0

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lr / 10)

    dummy_input = torch.randn(1, *ClassifyDataset.SHAPE)
    # 计算模型参数和FLOPs，拷贝一份model，因为profile会修改模型
    flops, params = thop.profile(copy.deepcopy(model), inputs=(dummy_input,))[0:2]
    writer.add_text("Profile", f"FLOPs: {flops/1e9:.2f}G, Params: {params/1e6:.2f}M")
    writer.add_text(
        "Config",
        f"Batch Size: {args.batch_size}, Learning Rate: {args.lr}, Seed: {args.seed}",
    )
    # 可视化模型结构
    writer.add_graph(model, (dummy_input,))

    trasform = ProbabilityTransform(  # noqa: F841
        [
            RandomChoiceTransform(
                [
                    enhance.GuassianNoise(std=partial(np.random.uniform, 5, 10)),
                    enhance.RandomNoise(low=-10, high=10),
                ]
            ),
            enhance.Flip(axis=0),
            enhance.Flip(axis=1),
            enhance.ScaleValue(scale=partial(np.random.uniform, 0.9, 1.1)),
        ],
        p=[1, 0.8, 0.8, 1],
    )

    val_dataset = ClassifyDataset(
        args.val,
        LABEL_LIST,
        # transform=trasform,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    train_dataset = ClassifyDataset(
        args.train,
        LABEL_LIST,
        # transform=trasform,
    )

    label_counts = np.zeros(len(LABEL_LIST))
    for label in train_dataset.labels:
        label_counts[label] += 1
    label_weights = 1 / label_counts

    label_weights /= label_weights.sum()
    sample_weights = label_weights[train_dataset.labels]
    sampler = WeightedRandomSampler(
        sample_weights, len(val_dataset) * 2, replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    max_metric = 0

    device = torch.device(args.device)
    model = model.to(device)

    def save_model(name: str, epoch: int = 0):
        if isinstance(writer, DummyWriter):
            return
        checkpoint_dir = osp.join(log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(
            {"model": model, "epoch": epoch},
            osp.join(checkpoint_dir, name),
        )

    def on_epoch_end(epoch: int):
        metric = validate(val_loader, model, device=device, epoch=epoch)
        nonlocal max_metric
        if metric > max_metric:
            max_metric = metric
            save_model("best.pth", epoch)
            print(f"Best Model Saved, Metric: {max_metric:.4f}")
        if epoch % 5 == 0 and epoch > 0 or epoch == begin_epoch + args.epochs - 1:
            save_model("latest.pth", epoch)
            print("Latest Model Saved")
        writer.add_scalar("Metric/Best", max_metric, epoch)

    if args.mode == "train":
        train(
            train_loader,
            model,
            optimizer,
            range(begin_epoch, begin_epoch + args.epochs),
            callback=on_epoch_end,
            device=device,
        )
    else:
        max_metric = validate(val_loader, model, device=device)
    print("Training Finished, Best Metric:", max_metric)


class DummyWriter(SummaryWriter):
    def __init__(self, save=True, *args, **kwargs):
        self.save = save

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        print(f"{tag}: {scalar_value}")

    def add_figure(self, tag, figure, global_step=None, close=True, walltime=None):
        if not self.save:
            return
        if not isinstance(figure, list):
            figure = [figure]
        os.makedirs(osp.join("figures", tag), exist_ok=True)
        for i, fig in enumerate(figure):
            fig.savefig(osp.join("figures", tag, f"{global_step}_{i}.png"))
            if close:
                plt.close(fig)

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        print(f"Text: {tag}\n{text_string}")

    def add_graph(self, model, input_to_model=None, verbose=False):
        print(f"Model: {model.__class__.__name__}")

    def flush(self):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-w", "--num_workers", type=int, default=5)
    parser.add_argument("-m", "--model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--train", type=str, default=osp.join("data", "train"))
    parser.add_argument("--val", type=str, default=osp.join("data", "val"))
    parser.add_argument("-n", "--name", type=str, default="")
    parser.add_argument(
        "mode", type=str, choices=["train", "val"], default="train", nargs="?"
    )

    args = parser.parse_args()

    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.name:
        name = f"{name}_{args.name}"
    log_dir = osp.join("runs", name)
    print(f"Log Dir: {log_dir}, Open Tensorboard with `tensorboard --logdir {log_dir}`")
    print("Open http://localhost:6006/ in your browser")
    # 强制设置seed，方便复现
    if args.seed is None:
        args.seed = random.randint(0, 1 << 32)
    seed_everything(args.seed)
    # 读取log_dir下的model.py文件
    if osp.exists(osp.join(log_dir, "model.py")):
        # 动态加载model.py
        print(f"Load 'model.py' from {log_dir}")
        model_module = importlib.import_module("model", package=log_dir)
        DASNet = model_module.DASNet
        ClassifyNet = model_module.ClassifyNet
    if args.mode == "train":
        writer = SummaryWriter(log_dir, flush_secs=10)
        # 备份model.py
        if not osp.exists(osp.join(log_dir, "model.py")):
            shutil.copy("model.py", osp.join(log_dir, "model.py"))
    else:
        writer = DummyWriter()

    main()
    writer.close()
