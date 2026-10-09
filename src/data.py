"""EMNIST ByClass loading, preprocessing, and label metadata."""

from __future__ import annotations

from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, random_split

CLASS_NAMES = tuple("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
NUM_CLASSES = len(CLASS_NAMES)


def emnist_transform():
    """Return the shared 32x32, correctly-oriented, [-1, 1] transform.

    torchvision exposes EMNIST glyphs transposed relative to their normal
    reading orientation. Rotating 90 degrees counter-clockwise and flipping
    horizontally corrects this before resizing and normalisation.
    """
    from torchvision import transforms
    from torchvision.transforms import functional as TF

    return transforms.Compose(
        [
            transforms.Lambda(lambda image: TF.hflip(TF.rotate(image, -90))),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def build_datasets(data_dir: str | Path, download: bool = True) -> tuple[Dataset, Dataset]:
    """Create official EMNIST ByClass train and test datasets."""
    from torchvision.datasets import EMNIST

    root = str(data_dir)
    transform = emnist_transform()
    return (
        EMNIST(root=root, split="byclass", train=True, download=download, transform=transform),
        EMNIST(root=root, split="byclass", train=False, download=download, transform=transform),
    )


def make_loaders(
    data_dir: str | Path,
    batch_size: int,
    validation_fraction: float,
    seed: int,
    num_workers: int = 0,
    download: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build deterministic train/validation splits and the official test loader."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    train_dataset, test_dataset = build_datasets(data_dir, download=download)
    validation_size = round(len(train_dataset) * validation_fraction)
    train_size = len(train_dataset) - validation_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, validation_subset = random_split(train_dataset, [train_size, validation_size], generator=generator)
    loader_options = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train_subset, shuffle=True, **loader_options),
        DataLoader(validation_subset, shuffle=False, **loader_options),
        DataLoader(test_dataset, shuffle=False, **loader_options),
    )


def label_name(label: int) -> str:
    return CLASS_NAMES[label]
