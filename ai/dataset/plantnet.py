
"""PlantNet dataset implementation.

This module provides a PyTorch Dataset for the PlantNet dataset.

Features:
- Automatic class discovery
- Recursive image discovery
- Deterministic class indexing
- pathlib-based filesystem operations
- PIL image loading
- Optional torchvision transforms
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import Dataset

from config import Config


class PlantNetDataset(Dataset[tuple[Tensor, int]]):
    """PyTorch Dataset for the PlantNet dataset."""

    _VALID_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }

    def __init__(
        self,
        config: Config,
        split: str,
        transform: Callable | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            config: Global project configuration.
            split: Dataset split ("train", "val", or "test").
            transform: Optional torchvision transform pipeline.

        Raises:
            ValueError: If an invalid split is provided.
            FileNotFoundError: If the dataset directory does not exist.
            RuntimeError: If no classes or images are found.
        """
        self.config = config
        self.transform = transform

        split_map = {
            "train": config.train_dir,
            "val": config.validation_dir,
            "test": config.test_dir,
        }

        if split not in split_map:
            raise ValueError(
                f"Invalid split '{split}'. "
                "Expected one of: train, val, test."
            )

        self.root = split_map[split]

        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root}"
            )

        self.classes = self._discover_classes()
        self.class_to_idx = {
            class_name: idx
            for idx, class_name in enumerate(self.classes)
        }
        self.idx_to_class = {
            idx: class_name
            for class_name, idx in self.class_to_idx.items()
        }

        self.samples = self._discover_samples()

        if not self.samples:
            raise RuntimeError(
                f"No images found inside '{self.root}'."
            )

    def _discover_classes(self) -> list[str]:
        """Discover all class folders.

        Returns:
            Sorted list of class names.

        Raises:
            RuntimeError: If no class directories are found.
        """
        classes = sorted(
            directory.name
            for directory in self.root.iterdir()
            if directory.is_dir()
        )

        if not classes:
            raise RuntimeError(
                f"No class folders found in {self.root}"
            )

        return classes

    def _discover_samples(self) -> list[tuple[Path, int]]:
        """Recursively discover all image files.

        Returns:
            List of (image_path, label_index) tuples.
        """
        samples: list[tuple[Path, int]] = []

        for class_name in self.classes:
            class_dir = self.root / class_name
            label = self.class_to_idx[class_name]

            for image_path in class_dir.rglob("*"):
                if (
                    image_path.is_file()
                    and image_path.suffix.lower()
                    in self._VALID_EXTENSIONS
                ):
                    samples.append((image_path, label))

        return samples

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Tensor, int]:
        """Return a single dataset sample.

        Args:
            index: Sample index.

        Returns:
            Tuple containing:
                image tensor
                integer label

        Raises:
            RuntimeError: If an image cannot be loaded.
        """
        image_path, label = self.samples[index]

        try:
            image = Image.open(image_path).convert("RGB")
        except (
            FileNotFoundError,
            UnidentifiedImageError,
            OSError,
        ) as exc:
            raise RuntimeError(
                f"Failed to load image: {image_path}"
            ) from exc

        if self.transform is not None:
            image = self.transform(image)

        return image, label

