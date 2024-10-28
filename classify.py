import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard.writer import SummaryWriter
import os
import os.path as osp
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
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, f1_score, confusion_matrix

from dataset import ClassifyDataset
import enhance
from enhance import RandomChoiceTransform, ProbabilityTransform
from model import DASNet, ClassifyNet
from utils import ReservoirSampler

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
    model: ClassifyNet,
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
            f"Epoch {epoch}, Loss: {total_loss/len(data_loader)}, Accuracy: {correct / total}"
        )
        writer.add_scalar("Loss/train", total_loss / len(data_loader), epoch)
        writer.add_scalar("Accuracy/train", correct / total, epoch)
        if callback:
            callback(epoch)
    writer.flush()


@torch.no_grad()
def validate(
    data_loader: DataLoader,
    model: ClassifyNet,
    device: torch.device | str = "cuda",
    epoch: int = 0,
    each_sample: int = 50,
):
    model = model.to(device)
    model.eval()
    all_preds = []
    ground_truths = []
    # 各类随机采样一部分数据用于可视化
    feature_samplers = [ReservoirSampler(each_sample) for _ in LABEL_LIST]
    # 各类随机采样一个数据用于CAM可视化激活图
    cam_samplers = [ReservoirSampler(1) for _ in LABEL_LIST]
    for inputs, labels in tqdm(data_loader, desc="Validating"):
        features = model.feature(inputs.to(device))
        # 采样的数据保存到列表中
        for i in range(len(labels)):
            label = labels[i].item()
            assert isinstance(label, int)
            feature_samplers[label].process(features[i])
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

    # TSNE可视化特征
    sample_features: list[torch.Tensor] = []
    for sampler in feature_samplers:
        samples = [sample.view(-1) for sample in sampler.samples]
        sample_features.extend(samples)
    feat_embed = TSNE(n_components=2, random_state=0).fit_transform(
        torch.stack(sample_features, dim=0).cpu().numpy()
    )
    fig, ax = plt.subplots()
    for i, label_str in enumerate(LABEL_LIST):
        start = i * each_sample
        end = (i + 1) * each_sample
        ax.scatter(feat_embed[start:end, 0], feat_embed[start:end, 1], label=label_str)
    ax.legend()
    plt.close(fig)
    writer.add_figure("TSNE", fig, global_step=epoch)

    # CAM可视化
    for truth, sampler in enumerate(cam_samplers):
        nimgs = len(LABEL_LIST) + 1
        nrows = int(nimgs**0.5)
        ncols = int(np.ceil(nimgs / nrows))
        fig, axes = plt.subplots(nrows, ncols)
        axes = axes.flatten()
        input_tensor, cam_tensor_list, predicted_tensor = sampler.samples[0]
        # input_tensor 对前5个通道维度求均值（低频通道），得到灰度图
        input_data = input_tensor[:5].mean(dim=0).cpu().numpy()
        predicted = predicted_tensor.item()
        # 对数变换，增强对比度
        input_data = np.abs(input_data, out=input_data)
        input_data = np.log1p(input_data, out=input_data)
        axes[0].imshow(
            input_data,
            cmap="jet",
            vmin=0,
            vmax=np.log1p(1000),
            aspect="auto",
            interpolation="bilinear",
        )
        axes[0].axis("off")
        input_idx = sampler.idxs[0]
        axes[0].set_title(f"Input {input_idx}")
        # 生成CAM
        vmax = max(cam.max().item() for cam in cam_tensor_list)
        vmin = min(cam.min().item() for cam in cam_tensor_list)
        for class_idx, cam_tensor in enumerate(cam_tensor_list):
            cam = cam_tensor.cpu().numpy()
            axes[class_idx + 1].imshow(
                cam,
                cmap="jet",
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
                interpolation="bilinear",
            )
            axes[class_idx + 1].axis("off")
            if class_idx == predicted:
                axes[class_idx + 1].set_title(f"{LABEL_LIST[class_idx]}(Predicted)")
            else:
                axes[class_idx + 1].set_title(f"{LABEL_LIST[class_idx]}")
        for i in range(len(cam_tensor_list) + 1, nrows * ncols):
            fig.delaxes(axes.flatten()[i])
        plt.tight_layout()
        plt.close(fig)
        writer.add_figure(f"CAM/{LABEL_LIST[truth]}", fig, global_step=epoch)

    writer.flush()
    return f1_weighted


def main():
    model = DASNet(ClassifyDataset.FREQ_LEN, len(LABEL_LIST))
    optimizer = optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-7)

    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location="cpu")
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

    dummy_input = torch.randn(1, *ClassifyDataset.SHAPE)
    # 计算模型参数和FLOPs，拷贝一份model，因为profile会修改模型
    flops, params = thop.profile(copy.deepcopy(model), inputs=(dummy_input,))[0:2]
    print(f"FLOPs: {flops/1e9:.2f}G, Params: {params/1e6:.2f}M")
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
        osp.join("data", "val"),
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
        osp.join("data", "train"),
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

    checkpoint_dir = osp.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    def save_model(name: str, epoch: int = 0):
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
            print("Best Model Saved")
        if epoch % 5 == 0 and epoch > 0 or epoch == begin_epoch + args.epochs - 1:
            save_model("latest.pth", epoch)
            print("Latest Model Saved")

    if args.epochs > 0:
        train(
            train_loader,
            model,
            optimizer,
            range(begin_epoch, begin_epoch + args.epochs),
            callback=on_epoch_end,
            device=device,
        )
    else:
        validate(val_loader, model, device=device)
    print("Training Finished, Best Metric:", max_metric)
    # # 用于可选的继续训练
    # breakpoint()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("name", type=str, default=None, nargs="?")

    args = parser.parse_args()

    if args.name is None:
        args.name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = osp.join("runs", "classify_" + args.name)
    print(f"Log Dir: {log_dir}, Open Tensorboard with `tensorboard --logdir {log_dir}`")
    print("Open http://localhost:6006/ in your browser")
    if args.seed is not None:
        seed_everything(args.seed)
    with SummaryWriter(log_dir, flush_secs=10) as writer:
        main()
