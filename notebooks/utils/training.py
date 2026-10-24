"""Обучающий цикл на чистом PyTorch.

Общий модуль для ноутбуков про CNN (цикл писали вручную в cnn_transfer.ipynb).
"""

import numpy as np
import torch
import tqdm.auto


def _computed_metrics(metric) -> dict:
    computed = metric.compute()
    if isinstance(computed, dict):
        return {key: value.item() for key, value in computed.items()}
    return {"accuracy": computed.item()}


def train_epoch(model, data_loader, criterion, optimizer, metric, device):
    model.train()
    metric.reset()
    losses = []
    pbar = tqdm.auto.tqdm(data_loader, desc="  train", leave=False, disable=None)
    for inputs, targets in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        preds = model(inputs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        metric.update(preds, targets)
        losses.append(loss.item())
        pbar.set_description(f"  train loss={losses[-1]:.4f}")
    metrics = {"loss": float(np.mean(losses))}
    metrics.update(_computed_metrics(metric))
    metric.reset()
    return metrics


@torch.no_grad()
def validate(model, data_loader, criterion, metric, device):
    model.eval()
    metric.reset()
    losses = []
    pbar = tqdm.auto.tqdm(data_loader, desc="  val  ", leave=False, disable=None)
    for inputs, targets in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        preds = model(inputs)
        loss = criterion(preds, targets)

        metric.update(preds, targets)
        losses.append(loss.item())
        pbar.set_description(f"  val   loss={losses[-1]:.4f}")
    metrics = {"loss": float(np.mean(losses))}
    metrics.update(_computed_metrics(metric))
    metric.reset()
    return metrics


def train(model, train_loader, val_loader, criterion, metric, optimizer, device, epochs=3, scheduler=None):
    history = {}
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, metric, device)
        val_metrics = validate(model, val_loader, criterion, metric, device)
        print(
            f"  train loss={train_metrics['loss']:.4f} | val loss={val_metrics['loss']:.4f}",
            "".join(f" | val {k}={v:.4f}" for k, v in val_metrics.items() if k != "loss"),
        )
        for key, value in train_metrics.items():
            history.setdefault(f"train_{key}", []).append(value)
        for key, value in val_metrics.items():
            history.setdefault(f"val_{key}", []).append(value)
        if scheduler is not None:
            scheduler.step()
    return history
