
from __future__ import annotations
"""Reusable utility functions for the Plant Identification project."""

import json
import logging
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def seed_everything(seed: int) -> None:
    """Seed all random number generators for reproducibility.

    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    state: dict[str, Any],
    filepath: str | Path,
) -> None:
    """Save a training checkpoint.

    Args:
        state: Checkpoint dictionary.
        filepath: Destination path.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(
    filepath: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint.

    Args:
        filepath: Checkpoint path.
        map_location: Torch map location.

    Returns:
        Loaded checkpoint.

    Raises:
        FileNotFoundError: If checkpoint is missing.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(filepath)

    checkpoint = torch.load(
        filepath,
        map_location=map_location,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary.")

    return checkpoint


def save_json(
    data: Any,
    filepath: str | Path,
) -> None:
    """Save JSON data.

    Dataclasses are automatically converted to dictionaries.

    Args:
        data: Serializable object.
        filepath: Output path.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if is_dataclass(data):
        data = asdict(data)

    with filepath.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )


def load_json(
    filepath: str | Path,
) -> Any:
    """Load JSON data.

    Args:
        filepath: JSON file.

    Returns:
        Parsed object.
    """
    filepath = Path(filepath)

    with filepath.open("r", encoding="utf-8") as file:
        return json.load(file)


class AverageMeter:
    """Track running averages."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset statistics."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(
        self,
        value: float,
        n: int = 1,
    ) -> None:
        """Update statistics.

        Args:
            value: New value.
            n: Number of samples.
        """
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


@torch.no_grad()
def accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Compute top-1 accuracy.

    Args:
        outputs: Model logits.
        targets: Ground-truth labels.

    Returns:
        Accuracy percentage.
    """
    predictions = outputs.argmax(dim=1)
    correct = (predictions == targets).sum().item()
    return 100.0 * correct / targets.size(0)


def get_device() -> torch.device:
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def create_logger(
    name: str,
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a reusable logger.

    Args:
        name: Logger name.
        log_file: Optional log file.
        level: Logging level.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_handler = logging.FileHandler(
            log_path,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def count_parameters(
    model: torch.nn.Module,
    trainable_only: bool = True,
) -> int:
    """Count model parameters.

    Args:
        model: PyTorch model.
        trainable_only: Count only trainable parameters.

    Returns:
        Number of parameters.
    """
    parameters = (
        p
        for p in model.parameters()
        if p.requires_grad or not trainable_only
    )

    return sum(p.numel() for p in parameters)


def print_gpu_info() -> None:
    """Print information about available CUDA devices."""
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    print(f"CUDA Version : {torch.version.cuda}")
    print(f"GPU Count    : {torch.cuda.device_count()}")

    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)

        print(f"\nGPU {index}")
        print(f"Name         : {props.name}")
        print(
            f"Memory       : "
            f"{props.total_memory / (1024 ** 3):.2f} GB"
        )
        print(
            f"Capability   : "
            f"{props.major}.{props.minor}"
        )


def save_training_plot(
    train_values: list[float],
    val_values: list[float],
    ylabel: str,
    filepath: str | Path,
) -> None:
    """Save a training curve.

    Args:
        train_values: Training metric history.
        val_values: Validation metric history.
        ylabel: Metric name.
        filepath: Output image path.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    epochs = range(1, len(train_values) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_values, label="Train")
    plt.plot(epochs, val_values, label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Epoch")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


# ------------------------------------------------------------------
# Evaluation Utilities
# ------------------------------------------------------------------
def top_k_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5,
) -> float:
    """Compute Top-k accuracy."""

    _, predicted = outputs.topk(
        k,
        dim=1,
        largest=True,
        sorted=True,
    )

    correct = predicted.eq(
        targets.view(-1, 1)
    )

    return (
        correct.any(dim=1)
        .float()
        .mean()
        .item()
        * 100.0
    )

def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
) -> dict[str, Any]:
    """Compute classification metrics and return the confusion matrix."""
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(y_true, y_pred)
    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "classification_report": report,
        "confusion_matrix": confusion,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


def save_confusion_matrix(
    confusion_matrix_values: np.ndarray,
    filepath: str | Path,
    class_names: list[str] | None = None,
) -> None:
    """Save a confusion matrix image."""

    filepath = Path(filepath)
    filepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(12, 10))

    plt.imshow(
        confusion_matrix_values,
        interpolation="nearest",
        cmap=plt.cm.Blues,
    )

    plt.title("Confusion Matrix")
    plt.colorbar()

    if (
        class_names is not None
        and len(class_names) <= 50
    ):
        ticks = np.arange(len(class_names))

        plt.xticks(
            ticks,
            class_names,
            rotation=90,
            fontsize=6,
        )

        plt.yticks(
            ticks,
            class_names,
            fontsize=6,
        )

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()

    plt.savefig(
        filepath,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()



def save_metrics_json(
    metrics: dict[str, Any],
    filepath: str | Path,
) -> None:
    """Save evaluation metrics as JSON."""

    metrics = metrics.copy()

    if "confusion_matrix" in metrics:
        metrics["confusion_matrix"] = (
            metrics["confusion_matrix"]
            .tolist()
        )

    save_json(
        metrics,
        filepath,
    )