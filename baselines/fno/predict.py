"""Inference script for generating predictions using FNO from neuraloperator."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from model import FNO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate predictions")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--test-data", type=str, required=True,
                        help="Path to test HDF5 file")
    parser.add_argument("--output", type=str, default="task1_pred.hdf5",
                        help="Path to output HDF5 file")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, cuda")

    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


def compute_rel_mse(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    """计算 Rel-MSE 指标。

    公式：
        - 逐时间步：rel_t = Σ(pred_t - gt_t)² / Σ(gt_t)²
        - 逐样本：对时间步取均值，上限 5.0
        - 最终：对所有样本取均值

    Args:
        pred: 预测值 [n_samples, n_steps, x_grid]
        gt: 真实值 [n_samples, n_steps, x_grid]
        eps: 数值稳定性

    Returns:
        Rel-MSE 值，0 表示完美预测
    """
    n_samples = pred.shape[0]
    sample_rel_mse = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        n_steps = pred.shape[1]
        rel_t = np.zeros(n_steps, dtype=np.float32)
        for t in range(n_steps):
            diff_sq = np.sum((pred[i, t] - gt[i, t]) ** 2)
            gt_sq = np.sum(gt[i, t] ** 2) + eps
            rel_t[t] = diff_sq / gt_sq
        sample_rel_mse[i] = min(np.mean(rel_t), 5.0)

    return float(np.mean(sample_rel_mse))


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config = checkpoint.get("config", {})
    else:
        state_dict = checkpoint
        config = {}

    # Build model
    model = FNO(
        n_modes=[config.get("n_modes", 16)],
        hidden_channels=config.get("hidden_channels", 64),
        in_channels=config.get("in_channels", 1),
        out_channels=config.get("out_channels", 1),
        lifting_channel_ratio=config.get("lifting_channel_ratio", 2.0),
        projection_channel_ratio=config.get("projection_channel_ratio", 2.0),
        n_layers=config.get("n_layers", 4),
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("Model loaded successfully")

    # Load test data
    print(f"Loading test data from {args.test_data}")
    with h5py.File(args.test_data, "r") as f:
        for key in ["tensor", "data", "u", "solution"]:
            if key in f:
                test_data = np.asarray(f[key])
                break
        else:
            for key in f:
                if isinstance(f[key], h5py.Dataset):
                    test_data = np.asarray(f[key])
                    break

    print(f"Test data shape: {test_data.shape}")

    # Get downsampling factors
    time_downsample = config.get("time_downsample", 5)
    space_downsample = config.get("space_downsample", 4)

    # Spatial downsampling
    if test_data.ndim == 3:
        test_data = test_data[:, :, ::space_downsample]
    elif test_data.ndim == 4:
        test_data = test_data[:, :, :, ::space_downsample]

    print(f"After spatial downsampling: {test_data.shape}")

    n_samples = test_data.shape[0]
    n_original_steps = test_data.shape[1]
    x_grid = test_data.shape[-1]

    # Time downsampling for model input
    # Original: 200 steps -> downsampled: 40 steps
    n_downsampled_steps = n_original_steps // time_downsample
    if n_original_steps % time_downsample == 0:
        n_downsampled_steps = n_original_steps // time_downsample
    else:
        n_downsampled_steps = n_original_steps // time_downsample + 1

    # Output: [N, 200, 256]
    output = np.zeros((n_samples, n_original_steps, x_grid), dtype=np.float32)

    print(f"Generating predictions for {n_samples} samples...")
    inference_start = time.time()

    with torch.no_grad():
        for i in range(n_samples):
            sample = test_data[i]
            if sample.ndim == 3:
                sample = sample.squeeze()

            # Time downsampling: interpolate from original to downsampled
            # Input: first 10 steps (original) -> downsampled to ~2 steps
            # We need to handle this carefully

            # Copy original time steps to output
            output[i, :10, :] = sample

            # Autoregressive prediction on downsampled time grid
            # Start from first downsampled time step
            downsampled_data = sample[::time_downsample, :]  # [40, 256]

            # Predict autoregressively
            pred_steps = len(downsampled_data)

            # Use first step as initial condition
            x = torch.as_tensor(downsampled_data[0:1, :], dtype=torch.float32, device=device)
            x = x.unsqueeze(0)  # [1, 1, 256]

            predictions = [downsampled_data[0]]

            for t in range(1, pred_steps):
                pred = model(x)  # [1, 1, 256]
                predictions.append(pred.squeeze().cpu().numpy())
                x = pred  # Use prediction as next input

            # Interpolate predictions back to original time grid
            pred_downsampled = np.stack(predictions, axis=0)  # [40, 256]

            # Linear interpolation to original 200 steps
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, len(pred_downsampled))
            x_new = np.linspace(0, 1, n_original_steps)

            pred_upsampled = np.zeros((n_original_steps, x_grid), dtype=np.float32)
            for j in range(x_grid):
                f = interp1d(x_old, pred_downsampled[:, j], kind="linear")
                pred_upsampled[:, j] = f(x_new)

            # Store predictions (skip first 10 input steps)
            output[i, 10:, :] = pred_upsampled[10:, :]

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{n_samples}")

    inference_time = time.time() - inference_start
    print(f"Inference time: {inference_time:.2f}s")

    # Compute Rel-MSE (steps 10:200)
    gt = test_data[:, 10:, :]  # Ground truth for prediction window
    pred_all = output[:, 10:, :]  # Predictions for same window
    rel_mse = compute_rel_mse(pred_all, gt)
    print(f"Rel-MSE: {rel_mse:.6f}")

    # Save output
    print(f"Saving predictions to {args.output}")
    import csv
    time_csv_path = Path(args.output).parent / "task1_time.csv"
    with open(time_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["train_time", "inference_time"])
        writer.writerow([0.0, inference_time])

    with h5py.File(args.output, "w") as f:
        f.create_dataset("data", data=output)
        f.attrs["rel_mse"] = rel_mse

    print("Done!")




if __name__ == "__main__":
    main()