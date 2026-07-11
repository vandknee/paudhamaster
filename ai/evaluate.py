"""Evaluate a trained MobileNetV3 model on the PlantNet dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset.plantnet import PlantNetDataset
from dataset.transforms import get_val_transforms
from models.mobilenet import get_model
from utils import (
    AverageMeter,
    accuracy,
    compute_classification_metrics,
    create_logger,
    get_device,
    save_confusion_matrix,
    save_metrics_json,
    top_k_accuracy,
)
logger = create_logger("evaluate")

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate a trained MobileNetV3 model."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the checkpoint file.",
    )

    return parser.parse_args()
def create_dataloader(
    config: Config,
) -> DataLoader:
    """Create the evaluation DataLoader."""

    dataset = PlantNetDataset(
        config=config,
        split="test",
        transform=get_val_transforms(config),
    )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate the model on the test dataset."""

    model.eval()

    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    criterion = nn.CrossEntropyLoss()

    y_true: list[int] = []
    y_pred: list[int] = []

    progress = tqdm(
        dataloader,
        desc="Evaluating",
    )

    for images, labels in progress:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        top1 = accuracy(
            outputs,
            labels,
        )

        top5 = top_k_accuracy(
            outputs,
            labels,
        )

        predictions = outputs.argmax(dim=1)

        y_true.extend(
            labels.cpu().tolist()
        )

        y_pred.extend(
            predictions.cpu().tolist()
        )

        loss_meter.update(
            loss.item(),
            images.size(0),
        )

        top1_meter.update(
            top1,
            images.size(0),
        )

        top5_meter.update(
            top5,
            images.size(0),
        )

        progress.set_postfix(
            loss=f"{loss_meter.avg:.4f}",
            top1=f"{top1_meter.avg:.2f}%",
            top5=f"{top5_meter.avg:.2f}%",
        )

    metrics = compute_classification_metrics(
        y_true,
        y_pred,
    )

    metrics["loss"] = loss_meter.avg
    metrics["accuracy"] = top1_meter.avg
    metrics["top5_accuracy"] = top5_meter.avg

    return metrics

def main() -> None:
    """Run the evaluation pipeline."""

    args = parse_args()

    config = Config()

    device = get_device()

    logger.info(
        "Using device: %s",
        device,
    )

    dataloader = create_dataloader(
        config,
    )

    model = get_model(
        config=config,
        checkpoint_path=args.checkpoint,
    )

    model = model.to(device)
    logger.info(
        "Loaded checkpoint: %s",
        args.checkpoint,
    )

    metrics = evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
    )

    confusion_matrix = metrics[
        "confusion_matrix"
    ]

    save_confusion_matrix(
        confusion_matrix_values=confusion_matrix,
        filepath=config.export_dir / "confusion_matrix.png",
        class_names=None,
    )

    config.export_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_metrics_json(
        metrics,
        config.export_dir / "evaluation_metrics.json",
    )

    logger.info("")

    logger.info(
        "Evaluation Results"
    )

    logger.info("-------------------------")

    logger.info(
        "Loss: %.4f",
        metrics["loss"],
    )

    logger.info(
        "Accuracy: %.2f%%",
        metrics["accuracy"],
    )

    logger.info(
        "Top-5 Accuracy: %.2f%%",
        metrics["top5_accuracy"],
    )

    logger.info(
        "Precision: %.4f",
        metrics["precision_macro"],
    )

    logger.info(
        "Recall: %.4f",
        metrics["recall_macro"],
    )

    logger.info(
        "F1 Score: %.4f",
        metrics["f1_macro"],
    )

    logger.info(
        "\nClassification Report\n%s",
        metrics["classification_report"],
    )

    logger.info(
        "Results saved to %s",
        config.export_dir,
    )

if __name__ == "__main__":
    main()