"""Prediction script for DeepONet on 1D Burgers equation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from baselines.deeponet.model import DeepONet1D
from baselines.deeponet.data_loader import get_project_root


def predict(
    checkpoint_path: str,
    initial_condition: np.ndarray,
    n_time: int = 40,
    n_space: int = 256,
    device: str = "cuda",
) -> np.ndarray:
    """Predict solution using trained DeepONet model.

    Args:
        checkpoint_path: path to model checkpoint
        initial_condition: initial condition at 256 spatial points [256]
        n_time: number of time steps (after downsampling)
        n_space: number of spatial points (after downsampling)
        device: device to run inference on

    Returns:
        solution: predicted solution [n_time, n_space]
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load model
    model = DeepONet1D(branch_dim=256, trunk_dim=2, width=50, branch_depth=2, trunk_depth=3)
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = get_project_root() / checkpoint_path

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Prepare inputs
    initial_condition = np.asarray(initial_condition, dtype=np.float32)
    if initial_condition.ndim == 1:
        initial_condition = initial_condition.reshape(1, -1)

    n_samples = initial_condition.shape[0]

    # Generate grid coordinates
    t = np.linspace(0, 1, n_time)
    x = np.linspace(0, 1, n_space)
    T, X = np.meshgrid(t, x, indexing="ij")
    grid = np.stack([T.ravel(), X.ravel()], axis=1).astype(np.float32)  # [10240, 2]

    n_grid_points = n_time * n_space

    # Expand for batch processing
    branch = torch.from_numpy(initial_condition).float().to(device)  # [n_samples, 256]
    trunk = torch.from_numpy(grid).float().to(device)  # [10240, 2]

    # Expand branch to match grid points
    branch_expanded = branch.unsqueeze(1).expand(-1, n_grid_points, -1)  # [n_samples, 10240, 256]
    trunk_expanded = trunk.unsqueeze(0).expand(n_samples, -1, -1)  # [n_samples, 10240, 2]

    # Concatenate inputs
    inputs = torch.cat([branch_expanded, trunk_expanded], dim=-1)  # [n_samples, 10240, 258]

    # Predict
    with torch.no_grad():
        inputs_flat = inputs.view(-1, 258)  # [n_samples * 10240, 258]
        pred = model(inputs_flat)  # [n_samples * 10240, 1]
        pred = pred.view(n_samples, n_grid_points)  # [n_samples, 10240]

    # Reshape to [n_samples, n_time, n_space]
    solution = pred.cpu().numpy().reshape(n_samples, n_time, n_space)

    if n_samples == 1:
        solution = solution[0]

    return solution


def predict_from_file(
    checkpoint_path: str,
    data_path: str,
    sample_indices: list[int] | None = None,
    time_ds: int = 5,
    space_ds: int = 4,
    device: str = "cuda",
) -> np.ndarray:
    """Predict solutions from HDF5 data file.

    Args:
        checkpoint_path: path to model checkpoint
        data_path: path to HDF5 data file
        sample_indices: indices of samples to predict (None = all)
        time_ds: time downsampling factor
        space_ds: spatial downsampling factor
        device: device to run inference on

    Returns:
        solutions: predicted solutions [n_samples, n_time, n_space]
    """
    import h5py

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load model
    model = DeepONet1D(branch_dim=256, trunk_dim=2, width=50, branch_depth=2, trunk_depth=3)
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = get_project_root() / checkpoint_path

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Load data
    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = get_project_root() / data_path

    with h5py.File(data_path, "r") as f:
        data = np.asarray(f["tensor"], dtype=np.float32)

    # Downsample
    data = data[:, ::time_ds, ::space_ds]
    n_samples, n_time, n_space = data.shape

    if sample_indices is not None:
        data = data[sample_indices]
        n_samples = len(sample_indices)

    # Get initial conditions
    initial_conditions = data[:, 0, :]  # [n_samples, 256]

    # Generate grid
    t = np.linspace(0, 1, n_time)
    x = np.linspace(0, 1, n_space)
    T, X = np.meshgrid(t, x, indexing="ij")
    grid = np.stack([T.ravel(), X.ravel()], axis=1).astype(np.float32)

    n_grid_points = n_time * n_space

    # Prepare tensors
    branch = torch.from_numpy(initial_conditions).float().to(device)
    trunk = torch.from_numpy(grid).float().to(device)

    branch_expanded = branch.unsqueeze(1).expand(-1, n_grid_points, -1)
    trunk_expanded = trunk.unsqueeze(0).expand(n_samples, -1, -1)

    inputs = torch.cat([branch_expanded, trunk_expanded], dim=-1)

    # Predict
    with torch.no_grad():
        inputs_flat = inputs.view(-1, 258)
        pred = model(inputs_flat)
        pred = pred.view(n_samples, n_grid_points)

    solutions = pred.cpu().numpy().reshape(n_samples, n_time, n_space)

    return solutions


def main():
    parser = argparse.ArgumentParser(description="Predict using trained DeepONet")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to HDF5 data file for batch prediction",
    )
    parser.add_argument(
        "--sample_indices",
        type=str,
        default=None,
        help="Comma-separated sample indices to predict",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save predictions (numpy format)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on",
    )

    args = parser.parse_args()

    if args.data_path is not None:
        sample_indices = None
        if args.sample_indices is not None:
            sample_indices = [int(i) for i in args.sample_indices.split(",")]

        predictions = predict_from_file(
            checkpoint_path=args.checkpoint,
            data_path=args.data_path,
            sample_indices=sample_indices,
            device=args.device,
        )

        print(f"Predictions shape: {predictions.shape}")

        if args.output_path is not None:
            output_path = Path(args.output_path)
            if not output_path.is_absolute():
                output_path = get_project_root() / output_path
            np.save(output_path, predictions)
            print(f"Predictions saved to {output_path}")
    else:
        print("Please provide --data_path for batch prediction")


if __name__ == "__main__":
    main()
