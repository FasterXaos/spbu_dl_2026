"""Хелперы для визуализаций.

Все функции **сохраняют** график в файл и не вызывают plt.show(),
чтобы картинки можно было генерить и из скриптов, и из ноутбуков.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def _prepare_dir(save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    return save_path


def denormalize(image: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    """Переводит нормализованный тензор (3, H, W) обратно в uint8-подобный массив [0, 1]."""
    img = image.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    return np.clip(img, 0, 1)


def show_image_batch(images, labels, classes, save_path, title=None, n=16, ncols=8, mean=None, std=None):
    """Сетка изображений батча с подписями классов. Сохраняет картинку в save_path."""
    if mean is not None:
        images = [denormalize(img, np.array(mean), np.array(std)) for img in images]
    else:
        images = [img.detach().cpu().numpy().transpose(1, 2, 0) for img in images]
    n = min(n, len(images))
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    for ax, img, label in zip(axes, images[:n], labels[:n]):
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(classes[label], fontsize=8)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    save_path = _prepare_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {save_path}")


def plot_train_history(history, title, save_path):
    """Графики лосса и точности по эпохам (train/val). Сохраняет картинку в save_path."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_loss"], "o-", label="train")
    ax1.plot(epochs, history["val_loss"], "s--", label="val")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_yscale("log")
    ax1.set_title(f"{title}: loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_accuracy"], "o-", label="train")
    ax2.plot(epochs, history["val_accuracy"], "s--", label="val")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_title(f"{title}: accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    save_path = _prepare_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {save_path}")


def plot_loss_curves(curves, title, save_path, xlabel="iteration", ylabel="loss", ylim=None):
    """Несколько кривых лосса на одном графике. curves: dict[name -> list of values]."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, values in curves.items():
        ax.plot(values, label=name, linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_path = _prepare_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {save_path}")


def plot_val_histories(histories, title, save_path, ylabel="val accuracy"):
    """Кривые одной метрики для нескольких обучений. histories: dict[name -> list of values]."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, values in histories.items():
        ax.plot(np.arange(1, len(values) + 1), values, "o-", label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_path = _prepare_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {save_path}")


def plot_feature_maps(activation, title, save_path, max_channels=16, ncols=8):
    """Карты активаций одного слоя для одного изображения.

    activation -- тензор (C, H, W); каждая карта нормализуется в [0, 1]
    (почему это нужно -- отдельный вопрос студентам: диапазоны у каналов разные,
    а imshow рисует значения из [0, 1]).
    """
    act = activation.detach().cpu().numpy()
    num_channels = min(max_channels, act.shape[0])
    ncols = min(ncols, num_channels)
    nrows = int(np.ceil(num_channels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.4 * ncols, 1.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[num_channels:]:
        ax.axis("off")
    for ax, channel in zip(axes, act[:num_channels]):
        channel = channel - channel.min()
        channel = channel / (channel.max() + 1e-8)
        ax.imshow(channel, cmap="coolwarm")
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    save_path = _prepare_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {save_path}")


def plot_bar_comparison(values, title, save_path, ylabel="accuracy"):
    """Столбчатая диаграмма для сравнения стратегий. values: dict[name -> number]."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    names = list(values.keys())
    scores = [values[n] for n in names]
    bars = ax.bar(names, scores, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(names))))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score, f"{score:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(scores) * 1.15)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    save_path = _prepare_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {save_path}")
