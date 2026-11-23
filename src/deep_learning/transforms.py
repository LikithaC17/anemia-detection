"""
Data Transformations and Augmentation Pipelines for Deep Learning
Author: Senior Deep Learning Engineer & PyTorch Developer
Project: AI-Based Anemia Detection
"""

from torchvision import transforms


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Returns data augmentation pipeline for training only.
    Preserves clinical color and anatomical integrity without aggressive distortions.
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-8, 8)),
        transforms.ColorJitter(brightness=0.08, contrast=0.08),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Returns deterministic transformation pipeline for validation and evaluation.
    No augmentation is applied.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
