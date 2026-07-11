
"""Global configuration for the Plant Identification project.

This module contains all configurable parameters used throughout the
training, evaluation, inference, and export pipelines.

Every other module should import the ``Config`` dataclass from here
instead of defining duplicate configuration values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch



@dataclass(slots=True)
class OptimizerConfig:
    """Configuration for the optimizer."""

    name: str = "AdamW"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass(slots=True)
class SchedulerConfig:
    """Configuration for the learning rate scheduler."""

    name: str = "CosineAnnealingLR"
    t_max: int = 30
    eta_min: float = 1e-6


@dataclass(slots=True)
class CheckpointConfig:
    """Checkpoint filename configuration."""

    best_model: str = "best_model.pt"
    last_model: str = "last_model.pt"
    optimizer_state: str = "optimizer.pt"
    scheduler_state: str = "scheduler.pt"


@dataclass(slots=True)
class Config:
    """Central configuration used across the entire project."""

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

        # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    _kaggle_root = Path(
        "/kaggle/input/datasets/emreekiz/plantnet-300k/plantnet_300K"
    )

    _local_root = Path(
        "dataset/plantnet_300K"
    )

    dataset_root: Path = (
        _kaggle_root / "images"
        if _kaggle_root.exists()
        else _local_root / "images"
    )

    train_folder: str = "train"
    validation_folder: str = "val"
    test_folder: str = "test"

    metadata_file: Path = (
        _kaggle_root / "plantnet300K_species_id_2_name.json"
        if _kaggle_root.exists()
        else _local_root / "plantnet300K_species_id_2_name.json"
    )

    image_metadata_file: Path = (
        _kaggle_root / "plantnet300K_metadata.json"
        if _kaggle_root.exists()
        else _local_root / "plantnet300K_metadata.json"
    )
    # ------------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------------
    checkpoint_dir: Path = Path("checkpoints")
    export_dir: Path = Path("exports")
    tensorboard_dir: Path = Path("runs")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_name: str = "mobilenet_v3_large"
    num_classes: int = 1081
    image_size: int = 224

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    epochs: int = 30
    batch_size: int = 64
    num_workers: int = 4

    seed: int = 42

    mixed_precision: bool = True

    # ------------------------------------------------------------------
    # Debug / Smoke Testing
    # ------------------------------------------------------------------
    debug_mode: bool = False

    debug_train_samples: int = 512
    debug_val_samples: int = 128

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # ------------------------------------------------------------------
    # Nested configurations
    # ------------------------------------------------------------------
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)

    # ------------------------------------------------------------------
    # Helper Properties
    # ------------------------------------------------------------------
    @property
    def train_dir(self) -> Path:
        """Return the training dataset directory."""
        return self.dataset_root / self.train_folder

    @property
    def validation_dir(self) -> Path:
        """Return the validation dataset directory."""
        return self.dataset_root / self.validation_folder

    @property
    def test_dir(self) -> Path:
        """Return the test dataset directory."""
        return self.dataset_root / self.test_folder

    def create_output_directories(self) -> None:
        """Create output directories if they do not already exist."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)


# Global configuration instance used across the project.
config = Config()

