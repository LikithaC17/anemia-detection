"""
DataLoader Factory for PyTorch Training and Evaluation
Author: Senior Deep Learning Engineer & PyTorch Developer
Project: AI-Based Anemia Detection
"""

from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader

from src.deep_learning.dataset import AnemiaDataset
from src.deep_learning.transforms import get_train_transforms, get_val_transforms


def create_dataloaders(
    train_manifest: str | Path,
    val_manifest: str | Path,
    test_manifest: Optional[str | Path] = None,
    modality: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = 224
) -> tuple[DataLoader, DataLoader, Optional[DataLoader], AnemiaDataset]:
    """
    Creates PyTorch DataLoaders for train, validation, and optional test splits.
    """
    train_tf = get_train_transforms(image_size)
    val_tf = get_val_transforms(image_size)

    train_dataset = AnemiaDataset(train_manifest, modality=modality, transform=train_tf)
    val_dataset = AnemiaDataset(val_manifest, modality=modality, transform=val_tf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    test_loader = None
    if test_manifest is not None:
        test_dataset = AnemiaDataset(test_manifest, modality=modality, transform=val_tf)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        )

    return train_loader, val_loader, test_loader, train_dataset
