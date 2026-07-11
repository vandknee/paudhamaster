"""Run inference using a trained MobileNetV3 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from config import Config
from dataset.transforms import get_val_transforms
from models.mobilenet import get_model
from utils import create_logger, get_device

logger = create_logger("predict")

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Predict plant species from images."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained checkpoint.",
    )

    parser.add_argument(
        "--image",
        type=Path,
        help="Predict a single image.",
    )

    parser.add_argument(
        "--folder",
        type=Path,
        help="Predict every image inside a folder.",
    )

    parser.add_argument(
        "--batch",
        nargs="+",
        type=Path,
        help="Predict multiple images.",
    )

    return parser.parse_args()

def load_species_mapping(
    filepath: Path,
) -> dict[str, str]:
    """Load PlantNet species ID to scientific name mapping."""

    if not filepath.exists():
        raise FileNotFoundError(
            f"Species mapping not found: {filepath}"
        )

    with filepath.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)
    

def load_image(
    image_path: Path,
    config: Config,
) -> torch.Tensor:
    """Load and preprocess a single image."""

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    try:
        image = Image.open(
            image_path
        ).convert("RGB")
    except (
        OSError,
        FileNotFoundError,
    ) as exc:
        raise RuntimeError(
            f"Failed to load image: {image_path}"
        ) from exc
    transform = get_val_transforms(
        config,
    )

    image = transform(image)

    return image.unsqueeze(0)

@torch.no_grad()
def predict_tensor(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device,
) -> tuple[list[int], list[float]]:
    """Return Top-5 predictions."""

    image = image.to(device)

    outputs = model(image)

    probabilities = torch.softmax(
        outputs,
        dim=1,
    )

    confidence, indices = torch.topk(
        probabilities,
        k=5,
        dim=1,
    )

    return (
        indices.squeeze(0).cpu().tolist(),
        confidence.squeeze(0).cpu().tolist(),
    )

def get_species_name(
    class_index: int,
    idx_to_class: dict[int, str],
    species_mapping: dict[str, str],
) -> str:
    """Convert a model class index to a scientific species name."""

    species_id = idx_to_class[class_index]

    return species_mapping.get(
        species_id,
        f"Unknown ({species_id})",
    )

def print_predictions(
    indices: list[int],
    confidences: list[float],
    idx_to_class: dict[int, str],
    species_mapping: dict[str, str],
) -> None:
    """Print Top-5 predictions."""

    print()

    print("=" * 60)
    print("Top-5 Predictions")
    print("=" * 60)

    for rank, (index, confidence) in enumerate(
        zip(indices, confidences),
        start=1,
    ):

        species = get_species_name(
            index,
            idx_to_class,
            species_mapping,
        )

        print(
            f"{rank}. "
            f"{species} "
            f"({confidence * 100:.2f}%)"
        )

    print("=" * 60)
    print()

def predict_image(
    image_path: Path,
    model: nn.Module,
    config: Config,
    device: torch.device,
    idx_to_class: dict[int, str],
    species_mapping: dict[str, str],
) -> None:
    """Predict a single image."""

    logger.info(
        "Predicting %s",
        image_path,
    )

    image = load_image(
        image_path,
        config,
    )

    indices, confidences = predict_tensor(
        model,
        image,
        device,
    )

    print_predictions(
        indices,
        confidences,
        idx_to_class,
        species_mapping,
    )

def predict_folder(
    folder: Path,
    model: nn.Module,
    config: Config,
    device: torch.device,
    idx_to_class: dict[int, str],
    species_mapping: dict[str, str],
) -> None:
    """Predict every image inside a folder."""
    if not folder.exists():
        raise FileNotFoundError(
            f"Folder not found: {folder}"
    )

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    images = sorted(
        image
        for image in folder.iterdir()
        if image.suffix.lower()
        in image_extensions
    )
    if not images:
        logger.warning(
            "No images found in %s",
            folder,
        )
        return

    for image in images:

        predict_image(
            image,
            model,
            config,
            device,
            idx_to_class,
            species_mapping,
        )
def predict_batch(
    image_paths: list[Path],
    model: nn.Module,
    config: Config,
    device: torch.device,
    idx_to_class: dict[int, str],
    species_mapping: dict[str, str],
) -> None:
    """Predict multiple images."""

    for image in image_paths:

        predict_image(
            image,
            model,
            config,
            device,
            idx_to_class,
            species_mapping,
        )
def build_idx_to_class(
    config: Config,
) -> dict[int, str]:
    """Build the class index mapping from the training folders."""

    train_root = (
    config.train_dir
)

    classes = sorted(
        directory.name
        for directory in train_root.iterdir()
        if directory.is_dir()
    )

    return {
        index: class_name
        for index, class_name
        in enumerate(classes)
    }

def main() -> None:
    """Run prediction."""

    args = parse_args()

    config = Config()

    device = get_device()

    logger.info(
        "Using device: %s",
        device,
    )

    model = get_model(
        config=config,
        checkpoint_path=args.checkpoint,
    )

    model = model.to(device)

    model.eval()

    logger.info(
        "Loaded checkpoint: %s",
        args.checkpoint,
    )

    species_mapping = load_species_mapping(
        config.metadata_file,
    )

    idx_to_class = build_idx_to_class(
        config,
    )

    if args.image is not None:

        predict_image(
            args.image,
            model,
            config,
            device,
            idx_to_class,
            species_mapping,
        )

        return

    if args.folder is not None:

        predict_folder(
            args.folder,
            model,
            config,
            device,
            idx_to_class,
            species_mapping,
        )

        return

    if args.batch is not None:

        predict_batch(
            args.batch,
            model,
            config,
            device,
            idx_to_class,
            species_mapping,
        )

        return

    raise ValueError(
        "Specify one of --image, --folder or --batch."
    )

if __name__ == "__main__":
    main()