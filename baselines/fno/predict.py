"""Inference script for generating predictions."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.models.fno import FNO1d, FNO1dTime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate predictions")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--test-data", type=str, required=True,
                        help="Path to test HDF5 file")
    parser.add_argument("--output", type=str, required=True,
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
        # 逐时间步计算
        n_steps = pred.shape[1]
        rel_t = np.zeros(n_steps, dtype=np.float32)
        for t in range(n_steps):
            diff_sq = np.sum((pred[i, t] - gt[i, t]) ** 2)
            gt_sq = np.sum(gt[i, t] ** 2) + eps
            rel_t[t] = diff_sq / gt_sq

        # 对时间步取均值，上限 5.0
        sample_rel_mse[i] = min(np.mean(rel_t), 5.0)

    return float(np.mean(sample_rel_mse))


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    model_type = config.get("model_type", "fno1d")
    if model_type == "fno1dtime":
        model = FNO1dTime(
            in_channels=config.get("input_steps", 10),
            out_channels=config.get("out_channels", 1),
            modes=config.get("modes", 12),
            width=config.get("width", 32),
            n_layers=config.get("n_layers", 4),
            n_steps=config.get("pred_steps", 190),
        ).to(device)
        print(f"Model: FNO1dTime")
    else:
        model = FNO1d(
            in_channels=config.get("input_steps", 10),
            out_channels=config.get("pred_steps", 190),
            modes=config.get("modes", 12),
            width=config.get("width", 32),
            n_layers=config.get("n_layers", 4),
        ).to(device)
        print(f"Model: FNO1d")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loading test data from {args.test_data}")
    with h5py.File(args.test_data, "r") as f:
        # Try common key names
        for key in ["tensor", "data", "u", "solution"]:
            if key in f:
                test_data = np.asarray(f[key])
                break
        else:
            # Find largest numeric dataset
            for key in f:
                if isinstance(f[key], h5py.Dataset):
                    test_data = np.asarray(f[key])
                    break

    print(f"Test data shape: {test_data.shape}")

    # Apply spatial downsampling
    reduced_resolution = config.get("reduced_resolution", 4)
    if test_data.ndim == 3:
        test_data = test_data[:, :, ::reduced_resolution]
    elif test_data.ndim == 4:
        test_data = test_data[:, :, :, ::reduced_resolution]

    input_steps = config.get("input_steps", 10)
    pred_steps = config.get("pred_steps", 190)

    n_samples = test_data.shape[0]
    x_grid = test_data.shape[-1]

    # Output array: [N, 200, 256]
    # First 10 steps from input, next 190 from prediction
    output = np.zeros((n_samples, 200, x_grid), dtype=np.float32)

    # Ground truth for Rel-MSE: [N, pred_steps, x_grid]
    gt = np.zeros((n_samples, pred_steps, x_grid), dtype=np.float32)

    print(f"Generating predictions for {n_samples} samples...")
    inference_start = time.time()

    with torch.no_grad():
        for i in range(n_samples):
            sample = test_data[i]
            if sample.ndim == 3:
                sample = sample.squeeze()

            # Input: first 10 time steps
            x = torch.as_tensor(sample[:input_steps], dtype=torch.float32, device=device)
            x = x.unsqueeze(0)  # [1, input_steps, x_grid]

            # Predict
            pred = model(x)  # [1, pred_steps, x_grid]
            pred = pred.squeeze(0).cpu().numpy()

            # Copy to output
            output[i, :input_steps, :] = sample[:input_steps]
            output[i, input_steps:, :] = pred

            # Store ground truth (time steps 10:200 from original data)
            gt[i] = sample[input_steps:input_steps + pred_steps]

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{n_samples}")

    inference_time = time.time() - inference_start
    print(f"Inference time: {inference_time:.2f}s")

    # Compute Rel-MSE
    pred_all = output[:, input_steps:, :]  # [N, pred_steps, x_grid]
    rel_mse = compute_rel_mse(pred_all, gt)
    print(f"Rel-MSE: {rel_mse:.6f}")

    # Save output
    print(f"Saving predictions to {args.output}")
    with h5py.File(args.output, "w") as f:
        f.create_dataset("pred", data=output)
        f.attrs["rel_mse"] = rel_mse

    print("Done!")


if __name__ == "__main__":
    main()
