"""Training script for FNO on 1D Burgers equation."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.burgers_dataset import BurgersDataset
from src.models.fno import FNO1d, FNO1dTime
from src.models import count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FNO on 1D Burgers")

    # Data
    parser.add_argument("--hdf5-path", type=str, required=True,
                        help="Path to HDF5 data file")
    parser.add_argument("--input-steps", type=int, default=10,
                        help="Number of input time steps")
    parser.add_argument("--pred-steps", type=int, default=190,
                        help="Number of prediction time steps")
    parser.add_argument("--reduced-resolution", type=int, default=4,
                        help="Spatial downsampling factor")
    parser.add_argument("--reduced-resolution-t", type=int, default=1,
                        help="Temporal downsampling factor")

    # Model
    parser.add_argument("--model-type", type=str, default="fno1d",
                        choices=["fno1d", "fno1dtime"],
                        help="Model type: fno1d (direct) or fno1dtime (autoregressive)")
    parser.add_argument("--modes", type=int, default=16,
                        help="Number of Fourier modes")
    parser.add_argument("--width", type=int, default=128,
                        help="Hidden layer width")
    parser.add_argument("--n-layers", type=int, default=4,
                        help="Number of Fourier layers")
    parser.add_argument("--out-channels", type=int, default=1,
                        help="Output channels per step (FNO1dTime only, usually 1)")

    # Training
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping max norm (0 to disable)")
    parser.add_argument("--train-split", type=float, default=0.9,
                        help="Training data fraction")

    # Output
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Device
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, cuda, cuda:0, etc.")

    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


class RelativeMSELoss(nn.Module):
    """Relative MSE loss aligned with evaluation metric.

    Computes per-timestep energy normalization:
        sum((pred - target)^2, dim=-1) / sum(target^2 + eps, dim=-1)
    then averages over batch and time dimensions.
    This matches the Rel-MSE evaluation metric.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sq_error = (pred - target) ** 2
        sq_target = target ** 2 + self.eps
        # Sum over spatial dimension (energy normalization per timestep)
        rel_per_step = sq_error.sum(dim=-1) / sq_target.sum(dim=-1)  # [batch, n_steps]
        return rel_per_step.mean()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # x: [batch, input_steps, x_grid]
        # y: [batch, pred_steps, x_grid]
        pred = model(x)

        loss = criterion(pred, y)
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = criterion(pred, y)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log file
    log_file = output_dir / "train.log"

    def log(msg: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line)
        with open(log_file, "a") as f:
            f.write(line + "\n")

    log(f"Arguments: {json.dumps(vars(args), indent=2)}")
    log(f"Device: {device}")

    # Dataset
    log("Loading dataset...")
    dataset = BurgersDataset(
        hdf5_path=args.hdf5_path,
        input_steps=args.input_steps,
        pred_steps=args.pred_steps,
        reduced_resolution=args.reduced_resolution,
        reduced_resolution_t=args.reduced_resolution_t,
    )
    log(f"Dataset size: {len(dataset)}")
    log(f"Sample shape: input={dataset.sample_shape}")

    # Get spatial grid size from a sample
    x_sample, y_sample = dataset[0]
    x_grid = x_sample.shape[-1]
    log(f"Spatial grid: {x_grid}")

    # Train/val split
    n_train = int(len(dataset) * args.train_split)
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    log(f"Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Model
    if args.model_type == "fno1dtime":
        model = FNO1dTime(
            in_channels=args.input_steps,
            out_channels=args.out_channels,
            modes=args.modes,
            width=args.width,
            n_layers=args.n_layers,
            n_steps=args.pred_steps,
        ).to(device)
        log(f"Model: FNO1dTime with {count_parameters(model):,} parameters")
    else:
        model = FNO1d(
            in_channels=args.input_steps,
            out_channels=args.pred_steps,
            modes=args.modes,
            width=args.width,
            n_layers=args.n_layers,
        ).to(device)
        log(f"Model: FNO1d with {count_parameters(model):,} parameters")

    # Optimizer and loss
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    criterion = RelativeMSELoss()

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float("inf")
    if args.checkpoint:
        log(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    # Training loop
    train_start_time = time.time()
    log(f"Starting training from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start

        log(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": vars(args),
            }, checkpoint_path)
            log(f"  -> Saved best model (val_loss={val_loss:.6f})")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch{epoch+1}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": vars(args),
            }, checkpoint_path)

    total_time = time.time() - train_start_time
    log(f"Training completed in {total_time/60:.1f} minutes")
    log(f"Best validation loss: {best_val_loss:.6f}")

    # Save final model
    final_path = output_dir / "final_model.pt"
    torch.save({
        "epoch": args.epochs - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "config": vars(args),
    }, final_path)
    log(f"Final model saved to {final_path}")


if __name__ == "__main__":
    main()
