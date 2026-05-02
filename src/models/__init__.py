"""Model utilities and factory functions."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .fno import FNO1d, FNO1dTime


def create_fno_model(
    config: dict[str, Any],
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Create FNO model from config dict.

    Args:
        config: Model configuration with keys:
            - in_channels: Number of input channels (time steps)
            - out_channels: Number of output channels (predicted steps)
            - modes: Number of Fourier modes
            - width: Hidden layer width
            - n_layers: Number of Fourier layers
            - n_steps: Number of autoregressive steps (for FNO1dTime)
            - padding: Optional zero-padding
        device: Target device

    Returns:
        Configured model on specified device
    """
    model_type = config.get("type", "FNO1dTime")

    if model_type == "FNO1d":
        model = FNO1d(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            modes=config["modes"],
            width=config["width"],
            n_layers=config.get("n_layers", 4),
            padding=config.get("padding", 0),
        )
    elif model_type == "FNO1dTime":
        model = FNO1dTime(
            in_channels=config["in_channels"],
            out_channels=config.get("out_channels", 1),
            modes=config["modes"],
            width=config["width"],
            n_layers=config.get("n_layers", 4),
            n_steps=config.get("n_steps", 1),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load model from checkpoint.

    Args:
        model: Model instance to load weights into
        checkpoint_path: Path to .pt file
        device: Target device

    Returns:
        Checkpoint dict with training state (if available)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return checkpoint
