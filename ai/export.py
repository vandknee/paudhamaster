"""Export trained MobileNetV3 models to deployment formats."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import onnx
import onnxruntime as ort
from onnxsim import simplify
from torch import nn

from config import Config
from models.mobilenet import get_model
from utils import create_logger, get_device

logger = create_logger("export")

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Export a trained model."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint to export.",
    )

    return parser.parse_args()

def create_dummy_input(
    config: Config,
    device: torch.device,
) -> torch.Tensor:
    """Create a dummy input tensor."""

    return torch.randn(
        1,
        3,
        config.image_size,
        config.image_size,
        device=device,
    )

def load_model(
    config: Config,
    checkpoint: Path,
    device: torch.device,
) -> nn.Module:
    """Load the trained model."""

    model = get_model(
        config=config,
        checkpoint_path=checkpoint,
    )

    model.to(device)

    model.eval()

    return model

def export_torchscript(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
) -> None:
    """Export the model to TorchScript."""

    logger.info(
        "Exporting TorchScript model..."
    )

    traced_model = torch.jit.trace(
        model,
        dummy_input,
    )

    traced_model.save(
        str(output_path)
    )

    logger.info(
        "TorchScript saved to %s",
        output_path,
    )

def export_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
) -> None:
    """Export the model to ONNX."""

    logger.info(
        "Exporting ONNX model..."
    )

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {
                0: "batch_size",
            },
            "output": {
                0: "batch_size",
            },
        },
    )

    logger.info(
        "ONNX saved to %s",
        output_path,
    )

def simplify_onnx(
    onnx_path: Path,
) -> None:
    """Simplify an ONNX model."""

    logger.info(
        "Simplifying ONNX model..."
    )

    model = onnx.load(
        str(onnx_path)
    )

    simplified_model, check = simplify(
        model,
    )

    if not check:
        raise RuntimeError(
            "ONNX simplification failed."
        )

    onnx.save(
        simplified_model,
        str(onnx_path),
    )

    logger.info(
        "ONNX simplification complete."
    )
def verify_onnx(
    onnx_path: Path,
    dummy_input: torch.Tensor,
) -> None:
    """Verify the exported ONNX model."""

    logger.info(
        "Verifying ONNX model..."
    )

    session = ort.InferenceSession(
        str(onnx_path),
        providers=[
            "CPUExecutionProvider",
        ],
    )

    input_name = (
        session
        .get_inputs()[0]
        .name
    )

    session.run(
        None,
        {
            input_name:
            dummy_input.cpu().numpy()
        },
    )

    logger.info(
        "ONNX verification successful."
    )

def main() -> None:
    """Export trained models."""

    args = parse_args()

    config = Config()

    config.create_output_directories()

    device = get_device()

    logger.info(
        "Using device: %s",
        device,
    )

    model = load_model(
        config,
        args.checkpoint,
        device,
    )

    dummy_input = create_dummy_input(
        config,
        device,
    )

    checkpoint_name = (
        args.checkpoint.stem
    )

    torchscript_path = (
        config.export_dir
        / f"{checkpoint_name}.ts"
    )

    onnx_path = (
        config.export_dir
        / f"{checkpoint_name}.onnx"
    )

    try:

        export_torchscript(
            model,
            dummy_input,
            torchscript_path,
        )

        export_onnx(
            model,
            dummy_input,
            onnx_path,
        )

        simplify_onnx(
            onnx_path,
        )

        verify_onnx(
            onnx_path,
            dummy_input,
        )

    except Exception:

        logger.exception(
            "Export failed."
        )

        raise

    logger.info(
        "Export completed successfully."
    )

    logger.info(
        "TorchScript : %s",
        torchscript_path,
    )

    logger.info(
        "ONNX        : %s",
        onnx_path,
    )

if __name__ == "__main__":
    main()