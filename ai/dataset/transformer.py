
"""Image preprocessing pipelines for the Plant Identification project.

This module provides torchvision transform pipelines for training and
validation/inference. All image preprocessing parameters are derived
from the project configuration.
"""

from __future__ import annotations

from torchvision import transforms

from config import Config

# ImageNet normalization statistics
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def get_train_transforms(config: Config) -> transforms.Compose:
    """Create the training image transformation pipeline.

    The training pipeline applies common data augmentations to improve
    model generalization while maintaining compatibility with
    MobileNetV3 pretrained ImageNet weights.

    Args:
        config: Project configuration.

    Returns:
        A torchvision Compose object containing the training transforms.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=config.image_size,
                scale=(0.8, 1.0),
                ratio=(0.75, 1.33),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def get_val_transforms(config: Config) -> transforms.Compose:
    """Create the validation/inference transformation pipeline.

    This pipeline performs deterministic preprocessing without data
    augmentation to ensure consistent evaluation results.

    Args:
        config: Project configuration.

    Returns:
        A torchvision Compose object containing the validation transforms.
    """
    return transforms.Compose(
        [
            transforms.Resize(
                (config.image_size, config.image_size)
            ),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )

