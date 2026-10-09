"""Save a labelled EMNIST sample grid for manual orientation verification."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src.data import build_datasets, label_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="runs/emnist_preview.png")
    parser.add_argument("--count", type=int, default=25)
    args = parser.parse_args()
    dataset, _ = build_datasets(args.data_dir)
    columns = 5
    rows = (args.count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(10, 2 * rows))
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index < args.count:
            image, label = dataset[index]
            axis.imshow(image.squeeze(), cmap="gray", vmin=-1, vmax=1)
            axis.set_title(f"{label}: {label_name(label)}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
