
"""MobileNetV3 model definition for the Plant Identification project."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    mobilenet_v3_large,
)

from config import Config


def _replace_classifier(model: nn.Module, num_classes: int) -> None:
    """Replace the MobileNetV3 classification head.

    Args:
        model: MobileNetV3 model.
        num_classes: Number of output classes.
    """
    if not hasattr(model, "classifier"):
        raise AttributeError(
            "The provided model does not expose a 'classifier' module."
        )

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)


def _set_backbone_trainable(
    model: nn.Module,
    trainable: bool,
) -> None:
    """Freeze or unfreeze the feature extractor.

    Args:
        model: MobileNetV3 model.
        trainable: Whether the backbone should be trainable.
    """
    for parameter in model.features.parameters():
        parameter.requires_grad = trainable


def _load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
) -> None:
    """Load model weights from a checkpoint.

    Supports checkpoints saved as either:
    - model.state_dict()
    - {"model_state_dict": ...}

    Args:
        model: Model instance.
        checkpoint_path: Path to the checkpoint file.

    Raises:
        FileNotFoundError:
            If the checkpoint does not exist.
        RuntimeError:
            If loading the checkpoint fails.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state_dict)


def get_model(
    config: Config,
    *,
    freeze_backbone: bool = False,
    checkpoint_path: Path | None = None,
) -> nn.Module:
    """Create a MobileNetV3 Large model.

    The returned model always remains on the CPU. Device placement is
    handled externally by the training or inference pipeline.

    Args:
        config: Project configuration.
        freeze_backbone: Whether to freeze the feature extractor.
        checkpoint_path: Optional checkpoint to load.

    Returns:
        Configured MobileNetV3 model.
    """
    model = mobilenet_v3_large(
        weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1
    )

    _replace_classifier(
        model=model,
        num_classes=config.num_classes,
    )

    _set_backbone_trainable(
        model=model,
        trainable=not freeze_backbone,
    )

    if checkpoint_path is not None:
        _load_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
        )

    return model