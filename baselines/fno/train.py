"""Training script for 1D Burgers equation using FNO from neuraloperator."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import FNO


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).resolve().parent.parent.parent



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FNO for 1D Burgers equation")

    parser.add_argument("--data", type=str, default="data/raw/1D_Burgers_Sols_Nu0.001.hdf5",
                        help="Path to training HDF5 file")
    parser.add_argument("--output", type=str, default="baselines/fno/checkpoints/fno_burgers_finetuned.pt",
                        help="Path to save checkpoint")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, cuda")

    # Model hyperparameters
    parser.add_argument("--n-modes", type=int, default=16,
                        help="Number of Fourier modes")
    parser.add_argument("--hidden-channels", type=int, default=64,
                        help="Hidden channel dimension")
    parser.add_argument("--in-channels", type=int, default=1,
                        help="Input channels")
    parser.add_argument("--out-channels", type=int, default=1,
                        help="Output channels")
    parser.add_argument("--lifting-channel-ratio", type=float, default=2.0,
                        help="Lifting channel ratio (relative to hidden_channels)")
    parser.add_argument("--projection-channel-ratio", type=float, default=2.0,
                        help="Projection channel ratio (relative to hidden_channels)")
    parser.add_argument("--n-layers", type=int, default=4,
                        help="Number of FNO layers")

    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")

    # Downsampling
    parser.add_argument("--pretrained", type=str, default="baselines/fno/checkpoints/1D_Burgers_Sols_Nu0.001_FNO.pt",
                        help="Path to pre-trained checkpoint for fine-tuning")
    parser.add_argument("--time-downsample", type=int, default=5,
                        help="Time downsampling factor")
    parser.add_argument("--space-downsample", type=int, default=4,
                        help="Spatial downsampling factor")

    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


def load_data(path: str, time_downsample: int, space_downsample: int):
    """Load and preprocess training data.

    Args:
        path: Path to HDF5 file
        time_downsample: Time downsampling factor (::5)
        space_downsample: Spatial downsampling factor (::4)

    Returns:
        input_tensor: [N, 1, x_grid] - first time step
        output_tensor: [N, 1, x_grid] - next time step
    """
    # Convert to absolute path if relative
    data_path = Path(path)
    if not data_path.is_absolute():
        data_path = get_project_root() / data_path

    print(f"Loading data from {data_path}")
    with h5py.File(data_path, "r") as f:
        # Data shape: [N, T, X] = [2048, 200, 1024]
        data = np.asarray(f["tensor"])

    print(f"Original data shape: {data.shape}")

    # Downsample: time ::5 (200->40), space ::4 (1024->256)
    data = data[:, ::time_downsample, ::space_downsample]
    print(f"Downsampled shape: {data.shape}")  # [2048, 40, 256]

    # Create input-output pairs: input = u(t), output = u(t+1)
    # For autoregressive training, we predict next time step
    n_samples = data.shape[0]
    n_steps = data.shape[1]
    x_grid = data.shape[2]

    # Total pairs: N * (T-1)
    total_pairs = n_samples * (n_steps - 1)

    inputs = np.zeros((total_pairs, 1, x_grid), dtype=np.float32)
    outputs = np.zeros((total_pairs, 1, x_grid), dtype=np.float32)

    idx = 0
    for i in range(n_samples):
        for t in range(n_steps - 1):
            inputs[idx, 0, :] = data[i, t, :]
            outputs[idx, 0, :] = data[i, t + 1, :]
            idx += 1

    print(f"Training pairs: {total_pairs}")
    return torch.from_numpy(inputs), torch.from_numpy(outputs)


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load data
    inputs, outputs = load_data(
        args.data,
        args.time_downsample,
        args.space_downsample
    )

    # Create DataLoader
    dataset = TensorDataset(inputs, outputs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Build model
    model = FNO(
        n_modes=[args.n_modes],
        hidden_channels=args.hidden_channels,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        lifting_channel_ratio=args.lifting_channel_ratio,
        projection_channel_ratio=args.projection_channel_ratio,
        n_layers=args.n_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Load pretrained weights if specified
    if args.pretrained:
        # Convert to absolute path if relative
        pretrained_path = Path(args.pretrained)
        if not pretrained_path.is_absolute():
            pretrained_path = get_project_root() / pretrained_path

        print(f"Loading pretrained weights from {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
        model.load_state_dict(state_dict, strict=False)
        print("Pretrained weights loaded")

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    # Loss function
    criterion = torch.nn.MSELoss()

    # Training loop
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

        avg_loss = total_loss / len(dataset)
        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{args.epochs} | Loss: {avg_loss:.6e} | Best: {best_loss:.6e} | LR: {scheduler.get_last_lr()[0]:.2e}")

    # Save checkpoint
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "n_modes": args.n_modes,
            "hidden_channels": args.hidden_channels,
            "in_channels": args.in_channels,
            "out_channels": args.out_channels,
            "lifting_channel_ratio": args.lifting_channel_ratio,
            "projection_channel_ratio": args.projection_channel_ratio,
            "n_layers": args.n_layers,
            "time_downsample": args.time_downsample,
            "space_downsample": args.space_downsample,
        }
    }
    torch.save(checkpoint, output_path)
    print(f"Checkpoint saved to {output_path}")


if __name__ == "__main__":
    main()