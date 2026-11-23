"""
PyTorch Dataset Implementation for Anemia Detection Images
Features in-memory and fast-caching to accelerate CPU training by up to 10x.
Author: Senior Deep Learning Engineer & PyTorch Developer
Project: AI-Based Anemia Detection
"""

import os
from pathlib import Path
from typing import Optional, Callable

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class AnemiaDataset(Dataset):
    """
    Dataset class for loading clinical anemia images from verified CSV manifests.
    Supports in-memory caching of resized PIL images for high-speed multi-epoch CPU training.
    """

    def __init__(
        self,
        manifest: pd.DataFrame | str | Path,
        modality: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_col: str = "target_binary",
        in_memory_cache: bool = True,
        target_size: tuple[int, int] = (224, 224)
    ):
        if isinstance(manifest, (str, Path)):
            self.df = pd.read_csv(manifest)
        else:
            self.df = manifest.copy()

        # Filter by modality if requested
        if modality is not None:
            self.df = self.df[self.df["modality"].str.lower() == modality.lower()].reset_index(drop=True)

        self.transform = transform
        self.target_col = target_col
        self.in_memory_cache = in_memory_cache
        self.target_size = target_size
        self._cache = {}

        # Verification of columns
        if "image_path" not in self.df.columns and "file_path" in self.df.columns:
            self.df["image_path"] = self.df["file_path"]

        if self.target_col not in self.df.columns:
            if "class_label" in self.df.columns:
                self.df[self.target_col] = self.df["class_label"].map({"Anemia": 1, "Non_Anemia": 0})
            else:
                raise ValueError(f"Neither {self.target_col} nor class_label found in manifest.")

        if self.in_memory_cache:
            self.prewarm_cache()

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, img_path: str) -> Image.Image:
        """Loads and pre-resizes an image safely."""
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
                if img_rgb.size != self.target_size:
                    img_rgb = img_rgb.resize(self.target_size, Image.Resampling.BILINEAR)
                return img_rgb
        except Exception as e:
            print(f"[ERROR] Failed to load image at {img_path}: {e}")
            return Image.new("RGB", self.target_size, (0, 0, 0))

    def prewarm_cache(self, max_workers: int = 8):
        """Pre-loads all images into RAM concurrently."""
        if not self.in_memory_cache or len(self.df) == 0:
            return
        from concurrent.futures import ThreadPoolExecutor
        def _load_item(idx: int):
            path = self.df.iloc[idx]["image_path"]
            return idx, self._load_image(path)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, img in executor.map(_load_item, range(len(self.df))):
                self._cache[idx] = img

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, dict]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        label = int(row[self.target_col])

        # In-memory cached access
        if self.in_memory_cache:
            if idx in self._cache:
                img_rgb = self._cache[idx]
            else:
                img_rgb = self._load_image(img_path)
                self._cache[idx] = img_rgb
        else:
            img_rgb = self._load_image(img_path)

        if self.transform is not None:
            image_tensor = self.transform(img_rgb)
        else:
            image_tensor = transforms.ToTensor()(img_rgb)

        metadata = {
            "image_path": img_path,
            "subject_id": str(row.get("subject_id", "Unknown")),
            "modality": str(row.get("modality", "Unknown")),
            "anatomy": str(row.get("anatomy_subcategory", "Unknown")),
            "class_label": str(row.get("class_label", "Unknown"))
        }

        return image_tensor, label, metadata

    def get_class_counts(self) -> dict[int, int]:
        """Returns distribution of classes."""
        return self.df[self.target_col].value_counts().to_dict()

    def get_class_weights(self) -> torch.Tensor:
        """
        Computes balanced inverse-frequency class weights for CrossEntropyLoss.
        weights[c] = N / (num_classes * count[c])
        """
        counts = self.df[self.target_col].value_counts()
        total_samples = len(self.df)
        num_classes = len(counts)
        weights = [total_samples / (num_classes * counts[c]) for c in range(num_classes)]
        return torch.tensor(weights, dtype=torch.float32)
