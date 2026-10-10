"""Plot loss and accuracy curves from an EMNIST training run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import yaml


def run_directory(config_path: Path, output_dir: str | None) -> Path:
    """Resolve a training run directory from its YAML configuration."""
    if output_dir:
        return Path(output_dir)
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {config_path}")
    with config_path.open() as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict) or "output_dir" not in config:
        raise ValueError("configuration must contain an output_dir setting")
    return Path(config["output_dir"])


def read_history(path: Path) -> dict[str, list[float]]:
    """Load the per-epoch values emitted by the training entry point."""
    if not path.is_file():
        raise FileNotFoundError(f"training history not found: {path}")
    with path.open(newline="") as history_file:
        rows = list(csv.DictReader(history_file))
    if not rows:
        raise ValueError(f"training history is empty: {path}")
    fields = ("epoch", "train_loss", "validation_loss", "validation_accuracy")
    missing = set(fields) - set(rows[0])
    if missing:
        raise ValueError(f"training history is missing columns: {', '.join(sorted(missing))}")
    return {field: [float(row[field]) for row in rows] for field in fields}


def plot_training_metrics(output_dir: Path, output: Path | None = None) -> Path:
    """Create a loss and accuracy figure for a completed or active run."""
    history = read_history(output_dir / "history.csv")
    metrics_path = output_dir / "test_metrics.json"
    test_accuracy: float | None = None
    if metrics_path.is_file():
        with metrics_path.open() as metrics_file:
            test_accuracy = float(json.load(metrics_file)["accuracy"])

    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = history["epoch"]
    loss_axis.plot(epochs, history["train_loss"], label="Train loss")
    loss_axis.plot(epochs, history["validation_loss"], label="Validation loss")
    loss_axis.set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    loss_axis.legend()
    loss_axis.grid(alpha=0.3)

    accuracy_axis.plot(epochs, history["validation_accuracy"], label="Validation accuracy")
    if test_accuracy is not None:
        accuracy_axis.axhline(test_accuracy, color="tab:green", linestyle="--", label=f"Test accuracy ({test_accuracy:.3f})")
    accuracy_axis.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    accuracy_axis.legend()
    accuracy_axis.grid(alpha=0.3)

    figure.suptitle(f"Training metrics: {output_dir}")
    figure.tight_layout()
    output = output or output_dir / "training_metrics.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot metrics from an EMNIST training run")
    parser.add_argument("--config", default="config.yaml", help="training YAML config used to find output_dir")
    parser.add_argument("--output-dir", help="run directory; overrides output_dir in --config")
    parser.add_argument("--output", help="PNG output path (default: <output-dir>/training_metrics.png)")
    args = parser.parse_args()

    output_dir = run_directory(Path(args.config), args.output_dir)
    output = Path(args.output) if args.output else output_dir / "training_metrics.png"
    plot_training_metrics(output_dir, output)
    print(f"saved {output}")
    if not (output_dir / "test_metrics.json").is_file():
        print("test_metrics.json not found; plotted validation accuracy only")


if __name__ == "__main__":
    main()
