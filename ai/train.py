
"""Training script for the Plant Identification project.

Part 1:
- Imports
- Argument parsing
- Helper functions
- Training loop
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import (
    DataLoader,
    Subset,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from dataset.plantnet import PlantNetDataset
from dataset.transforms import (
    get_train_transforms,
    get_val_transforms,
)
from models.mobilenet import get_model
from utils import (
    AverageMeter,
    accuracy,
    count_parameters,
    create_logger,
    get_device,
    load_checkpoint,
    print_gpu_info,
    save_checkpoint,
    seed_everything,
)

logger = create_logger("train")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train MobileNetV3 on PlantNet-300K."
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a checkpoint to resume training.",
    )

    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the MobileNet backbone.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs.",
    )

    return parser.parse_args()


def create_dataloaders(
    config: Config,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders.

    Args:
        config: Project configuration.

    Returns:
        Training and validation dataloaders.
    """
    train_dataset = PlantNetDataset(
        config=config,
        split="train",
        transform=get_train_transforms(config),
    )

    val_dataset = PlantNetDataset(
        config=config,
        split="val",
        transform=get_val_transforms(config),
    )

    if config.debug_mode:

        logger.warning(
            "DEBUG MODE ENABLED"
        )

    train_dataset = Subset(
        train_dataset,
        range(
            min(
                config.debug_train_samples,
                len(train_dataset),
            )
        ),
    )

    val_dataset = Subset(
        val_dataset,
        range(
            min(
                config.debug_val_samples,
                len(val_dataset),
            )
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )

    return train_loader, val_loader


def create_optimizer(
    model: nn.Module,
    config: Config,
) -> AdamW:
    """Create the optimizer.

    Args:
        model: Model to optimize.
        config: Project configuration.

    Returns:
        Configured AdamW optimizer.
    """
    return AdamW(
        params=filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        betas=config.optimizer.betas,
        eps=config.optimizer.eps,
    )


def create_scheduler(
    optimizer: AdamW,
    config: Config,
) -> CosineAnnealingLR:
    """Create the learning rate scheduler.

    Args:
        optimizer: Optimizer instance.
        config: Project configuration.

    Returns:
        Cosine Annealing scheduler.
    """
    return CosineAnnealingLR(
        optimizer=optimizer,
        T_max=config.scheduler.t_max,
        eta_min=config.scheduler.eta_min,
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    scaler: GradScaler,
    device: torch.device,
    config: Config,
) -> tuple[float, float]:
    """Train the model for one epoch.

    Args:
        model: Model being trained.
        dataloader: Training dataloader.
        criterion: Loss function.
        optimizer: Optimizer.
        scaler: Gradient scaler for AMP.
        device: Training device.
        config: Project configuration.

    Returns:
        Tuple containing:
            average training loss
            average training accuracy
    """
    model.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    progress = tqdm(
        dataloader,
        desc="Training",
        leave=False,
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

        optimizer.zero_grad(set_to_none=True)

        with autocast(
            enabled=(
                config.mixed_precision
                and device.type == "cuda"
        )
):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_accuracy = accuracy(
            outputs.detach(),
            labels,
        )

        loss_meter.update(
            loss.item(),
            images.size(0),
        )

        acc_meter.update(
            batch_accuracy,
            images.size(0),
        )

        progress.set_postfix(
            loss=f"{loss_meter.avg:.4f}",
            acc=f"{acc_meter.avg:.2f}%",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    return loss_meter.avg, acc_meter.avg


class EarlyStopping:
    """Early stopping utility.

    Training stops when the validation loss has not improved for
    a specified number of consecutive epochs.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
    ) -> None:
        """Initialize the early stopping monitor.

        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum decrease in validation loss to qualify
                as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, validation_loss: float) -> bool:
        """Update the monitor.

        Args:
            validation_loss: Current validation loss.

        Returns:
            True if the validation loss improved.
        """
        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
            return True

        self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

        return False


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: Config,
) -> tuple[float, float]:
    """Evaluate the model.

    Args:
        model: Model to evaluate.
        dataloader: Validation dataloader.
        criterion: Loss function.
        device: Device.
        config: Project configuration.

    Returns:
        Tuple containing:
            validation loss
            validation accuracy
    """
    model.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    progress = tqdm(
        dataloader,
        desc="Validation",
        leave=False,
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

        with autocast(
            enabled=(
                config.mixed_precision
                and device.type == "cuda"
            )
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        batch_accuracy = accuracy(outputs, labels)

        loss_meter.update(
            loss.item(),
            images.size(0),
        )

        acc_meter.update(
            batch_accuracy,
            images.size(0),
        )

        progress.set_postfix(
            loss=f"{loss_meter.avg:.4f}",
            acc=f"{acc_meter.avg:.2f}%",
        )

    return loss_meter.avg, acc_meter.avg


def resume_training(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    scaler: GradScaler,
) -> tuple[int, float]:
    """Resume training from a checkpoint.

    The checkpoint is expected to contain:

    - epoch
    - best_val_loss
    - model_state_dict
    - optimizer_state_dict
    - scheduler_state_dict
    - scaler_state_dict

    Args:
        checkpoint_path: Path to checkpoint.
        model: Model.
        optimizer: Optimizer.
        scheduler: Scheduler.
        scaler: AMP gradient scaler.

    Returns:
        Tuple containing:
            starting epoch
            best validation loss
    """
    logger.info(
        "Loading checkpoint: %s",
        checkpoint_path,
    )

    checkpoint = load_checkpoint(checkpoint_path)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

    start_epoch = checkpoint.get("epoch", 0) + 1

    best_loss = checkpoint.get(
        "best_val_loss",
        float("inf"),
    )

    logger.info(
        "Resumed training from epoch %d.",
        start_epoch,
    )

    return start_epoch, best_loss


def save_training_checkpoint(
    filepath: Path,
    epoch: int,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    scaler: GradScaler,
    best_val_loss: float,
) -> None:
    """Save a complete training checkpoint.

    Args:
        filepath: Destination file.
        epoch: Current epoch.
        model: Model.
        optimizer: Optimizer.
        scheduler: Scheduler.
        scaler: AMP scaler.
        best_val_loss: Best validation loss.
    """
    checkpoint = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }

    save_checkpoint(
        checkpoint,
        filepath,
    )

    logger.info(
        "Checkpoint saved to %s",
        filepath,
    )




def main() -> None:
    """Run the training pipeline."""
    args = parse_args()

    config = Config()

    if args.epochs is not None:
        config.epochs = args.epochs

    config.create_output_directories()

    seed_everything(config.seed)

    device = get_device()

    logger.info("Using device: %s", device)

    if device.type == "cuda":
        print_gpu_info()

    train_loader, val_loader = create_dataloaders(config)

    model = get_model(
        config=config,
        freeze_backbone=args.freeze_backbone,
    )

    logger.info(
        "Trainable parameters: %s",
        f"{count_parameters(model):,}",
    )

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer(
        model,
        config,
    )

    scheduler = create_scheduler(
        optimizer,
        config,
    )

    scaler = GradScaler(
        enabled=(
            config.mixed_precision
            and device.type == "cuda"
        )
    )

    writer = SummaryWriter(
        log_dir=config.tensorboard_dir
    )

    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume is not None:
        start_epoch, best_val_loss = resume_training(
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )

    early_stopping = EarlyStopping(
        patience=10,
    )

    logger.info("Starting training...")

    for epoch in range(
        start_epoch,
        config.epochs,
    ):
        logger.info(
            "Epoch [%d/%d]",
            epoch + 1,
            config.epochs,
        )

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            config=config,
        )

        val_loss, val_acc = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            config=config,
        )

        scheduler.step()

        writer.add_scalar(
            "Loss/train",
            train_loss,
            epoch,
        )

        writer.add_scalar(
            "Loss/validation",
            val_loss,
            epoch,
        )

        writer.add_scalar(
            "Accuracy/train",
            train_acc,
            epoch,
        )

        writer.add_scalar(
            "Accuracy/validation",
            val_acc,
            epoch,
        )

        writer.add_scalar(
            "Learning Rate",
            optimizer.param_groups[0]["lr"],
            epoch,
        )

        logger.info(
            (
                "Train Loss: %.4f | "
                "Train Acc: %.2f%% | "
                "Val Loss: %.4f | "
                "Val Acc: %.2f%%"
            ),
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        latest_checkpoint = (
            config.checkpoint_dir
            / config.checkpoints.last_model
        )

        save_training_checkpoint(
            filepath=latest_checkpoint,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_val_loss=best_val_loss,
        )

        improved = early_stopping.step(
            val_loss
        )

        if improved:
            best_val_loss = val_loss

            best_checkpoint = (
                config.checkpoint_dir
                / config.checkpoints.best_model
            )

            save_training_checkpoint(
                filepath=best_checkpoint,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_val_loss=best_val_loss,
            )

            logger.info(
                "New best model saved."
            )

        if early_stopping.should_stop:
            logger.info(
                "Early stopping triggered."
            )
            break

    writer.flush()
    writer.close()

    logger.info("Training complete.")


if __name__ == "__main__":
    main()

