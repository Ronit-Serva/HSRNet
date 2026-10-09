"""Configurable EMNIST ByClass training and evaluation entry point."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from .data import CLASS_NAMES, NUM_CLASSES, make_loaders
from .models import LeNet5


@dataclass
class TrainConfig:
    data_dir: str = "data"
    output_dir: str = "runs/emnist_lenet5"
    seed: int = 42
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    validation_fraction: float = 0.1
    num_workers: int = 0
    device: str = "auto"
    resume: str | None = None
    download: bool = True


def load_config(config_path: str | Path, overrides: dict[str, Any] | None = None) -> TrainConfig:
    """Load recognised training settings from YAML and apply non-null overrides."""
    import yaml

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"configuration file not found: {path}")
    with path.open() as config_file:
        values = yaml.safe_load(config_file) or {}
    if not isinstance(values, dict):
        raise ValueError("configuration must be a YAML mapping")
    known_keys = set(TrainConfig.__dataclass_fields__)
    unknown_keys = set(values) - known_keys
    if unknown_keys:
        raise ValueError(f"unknown configuration keys: {', '.join(sorted(unknown_keys))}")
    if overrides:
        values.update({key: value for key, value in overrides.items() if value is not None})
    return TrainConfig(**values)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def run_epoch(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_examples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_examples += labels.numel()
            total_loss += loss.item() * labels.numel()
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
    return {"loss": total_loss / total_examples, "accuracy": total_correct / total_examples}


def evaluate_with_confusion(
    model: nn.Module, loader: Iterable[tuple[Tensor, Tensor]], device: torch.device
) -> tuple[dict[str, float], torch.Tensor]:
    criterion = nn.CrossEntropyLoss()
    metrics = run_epoch(model, loader, criterion, device)
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images.to(device, non_blocking=True)).argmax(dim=1).cpu()
            for target, prediction in zip(labels, predictions):
                confusion[target.long(), prediction.long()] += 1
    return metrics, confusion


def save_checkpoint(path: Path, model: LeNet5, optimizer: Optimizer, epoch: int, best_validation_accuracy: float, config: TrainConfig) -> None:
    torch.save(
        {
            "model_name": "LeNet5",
            "num_classes": NUM_CLASSES,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_validation_accuracy": best_validation_accuracy,
            "config": asdict(config),
            "class_names": CLASS_NAMES,
        },
        path,
    )


def train(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    train_loader, validation_loader, test_loader = make_loaders(
        config.data_dir, config.batch_size, config.validation_fraction, config.seed, config.num_workers, config.download
    )
    model = LeNet5(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    start_epoch, best_validation_accuracy = 1, float("-inf")
    if config.resume:
        checkpoint = torch.load(config.resume, map_location=device, weights_only=False)
        if checkpoint["num_classes"] != NUM_CLASSES:
            raise ValueError("checkpoint does not contain a 62-class EMNIST model")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_validation_accuracy = checkpoint["best_validation_accuracy"]

    history_path = output_dir / "history.csv"
    write_header = not history_path.exists() or start_epoch == 1
    with history_path.open("a", newline="") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=["epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy"])
        if write_header:
            writer.writeheader()
        for epoch in range(start_epoch, config.epochs + 1):
            train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
            validation_metrics = run_epoch(model, validation_loader, criterion, device)
            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"validation_{k}": v for k, v in validation_metrics.items()}}
            writer.writerow(row)
            history_file.flush()
            if validation_metrics["accuracy"] > best_validation_accuracy:
                best_validation_accuracy = validation_metrics["accuracy"]
                save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, best_validation_accuracy, config)
            save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, best_validation_accuracy, config)
            print(f"epoch {epoch:03d}/{config.epochs}: train acc={train_metrics['accuracy']:.4f}, val acc={validation_metrics['accuracy']:.4f}")

    best_checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    test_metrics, confusion = evaluate_with_confusion(model, test_loader, device)
    (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2) + "\n")
    np.savetxt(output_dir / "confusion_matrix.csv", confusion.numpy(), fmt="%d", delimiter=",")
    class_totals = confusion.sum(dim=1).clamp_min(1)
    per_class = {name: float(confusion[index, index] / class_totals[index]) for index, name in enumerate(CLASS_NAMES)}
    (output_dir / "per_class_accuracy.json").write_text(json.dumps(per_class, indent=2) + "\n")
    print(f"test loss={test_metrics['loss']:.4f}, test accuracy={test_metrics['accuracy']:.4f}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train LeNet-5 on EMNIST ByClass")
    parser.add_argument("--config", default="config.yaml", help="YAML experiment configuration")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", help="auto, cpu, cuda, or a torch device string")
    parser.add_argument("--resume")
    parser.add_argument("--no-download", action="store_true", default=None)
    args = parser.parse_args()
    overrides = vars(args)
    config_path = overrides.pop("config")
    no_download = overrides.pop("no_download")
    if no_download:
        overrides["download"] = False
    return load_config(config_path, overrides)


if __name__ == "__main__":
    train(parse_args())
