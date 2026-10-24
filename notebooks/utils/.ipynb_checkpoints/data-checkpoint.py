"""Загрузка Oxford-IIIT Pet и создание даталоадеров.

Общий модуль для ноутбука cnn_transfer и скриптов из vis/.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

# Пайплайн без аугментаций: для валидации, теста и извлечения признаков
base_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

# Пайплайн с аугментациями: только для обучения
augmented_transform = transforms.Compose(
    [
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def make_dataloaders(
    data_root=None,
    batch_size: int = 64,
    train_transform=None,
    num_workers: int = 0,
    val_fraction: float = 0.2,
    seed: int = 123,
):
    """Возвращает (train_loader, val_loader, test_loader, num_classes, classes).
    """
    if data_root is None:
        data_root = Path(__file__).resolve().parents[1] / "data"
    train_transform = train_transform or base_transform

    train_dataset = OxfordIIITPet(
        root=data_root, split="trainval", target_types="category",
        transform=train_transform, download=True,
    )
    val_dataset = OxfordIIITPet(
        root=data_root, split="trainval", target_types="category",
        transform=base_transform, download=True,
    )
    test_dataset = OxfordIIITPet(
        root=data_root, split="test", target_types="category",
        transform=base_transform, download=True,
    )

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(train_dataset), generator=generator).tolist()
    n_train = int(len(train_dataset) * (1 - val_fraction))
    train_subset = Subset(train_dataset, permutation[:n_train])
    val_subset = Subset(val_dataset, permutation[n_train:])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    num_classes = len(train_dataset.classes)
    return train_loader, val_loader, test_loader, num_classes, train_dataset.classes
