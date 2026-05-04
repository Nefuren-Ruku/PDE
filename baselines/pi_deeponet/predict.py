"""Prediction script for PI-DeepONet (batched, submits format)."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from baselines.pi_deeponet.model import PIDeepONet1D
from baselines.deeponet.data_loader import get_project_root


def upsample_time(data, orig_t=200):
    n_samples, n_time_down, x_grid = data.shape
    t_down = np.linspace(0, 1, n_time_down)
    t_up = np.linspace(0, 1, orig_t)
    data_2d = data.transpose(0, 2, 1).reshape(-1, n_time_down)
    f = interp1d(t_down, data_2d, kind="linear", axis=1)
    result_2d = f(t_up)
    return result_2d.reshape(n_samples, x_grid, orig_t).transpose(0, 2, 1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="PI‑DeepONet inference")
    parser.add_argument("--checkpoint", type=str,
                        default="baselines/pi_deeponet/checkpoints/pi_deeponet_best.pt")
    parser.add_argument("--test-data", type=str,
                        default="data/raw/task1_test.hdf5")
    parser.add_argument("--output", type=str, default="task1_pred.hdf5")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--time-ds", type=int, default=5)
    parser.add_argument("--space-ds", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    print(f"Using device: {device}")

    project_root = get_project_root()
    ckpt_path = Path(args.checkpoint) if Path(args.checkpoint).is_absolute() else project_root / args.checkpoint
    test_path = Path(args.test_data) if Path(args.test_data).is_absolute() else project_root / args.test_data

    with h5py.File(test_path, "r") as f:
        raw_data = np.asarray(f["tensor"], dtype=np.float32)
    n_samples = raw_data.shape[0]

    if raw_data.shape[-1] == 1024:
        data_ds = raw_data[:, :, ::args.space_ds]
    else:
        data_ds = np.copy(raw_data)
    true_ic = data_ds[:, :10, :].astype(np.float32)

    n_time_ds = 40
    n_space = data_ds.shape[-1]
    branch_dim = n_space
    trunk_dim = 2
    t = np.linspace(0, 1, n_time_ds)
    x = np.linspace(0, 1, n_space)
    T, X = np.meshgrid(t, x, indexing="ij")
    trunk_grid = np.stack([T.ravel(), X.ravel()], axis=1).astype(np.float32)
    n_points = n_time_ds * n_space

    branch_input = data_ds[:, 0, :]

    model = PIDeepONet1D().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    branch_t = torch.from_numpy(branch_input).float().to(device)
    trunk_t = torch.from_numpy(trunk_grid).float().to(device)

    all_pred_ds = []
    bs = args.batch_size
    n_batches = int(np.ceil(n_samples / bs))

    t0 = time.time()
    with torch.no_grad():
        for i in range(n_batches):
            start = i * bs
            end = min(start + bs, n_samples)
            b = end - start

            batch_branch = branch_t[start:end]
            branch_exp = batch_branch.unsqueeze(1).expand(-1, n_points, -1)
            trunk_exp = trunk_t.unsqueeze(0).expand(b, -1, -1)
            inp = torch.cat([branch_exp, trunk_exp], dim=-1)
            inp_flat = inp.reshape(-1, branch_dim + trunk_dim)

            pred_flat = model(inp_flat)
            pred_batch = pred_flat.view(b, n_points).cpu().numpy()
            pred_batch = pred_batch.reshape(b, n_time_ds, n_space)
            all_pred_ds.append(pred_batch)

    inf_time = time.time() - t0
    pred_ds = np.concatenate(all_pred_ds, axis=0)
    pred_up = upsample_time(pred_ds, orig_t=200)
    pred_up[:, :10, :] = true_ic

    out_path = Path(args.output) if Path(args.output).is_absolute() else project_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("data", data=pred_up)
    print(f"Predictions saved to {out_path}")

    # 计时文件
    train_time = 0.0
    train_csv = ckpt_path.parent / "train_time.csv"
    if train_csv.exists():
        with open(train_csv, "r") as f:
            reader = csv.reader(f)
            next(reader)
            row = next(reader, None)
            if row:
                train_time = float(row[0])
    time_csv = out_path.parent / "task1_time.csv"
    with open(time_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["train_time", "inference_time"])
        writer.writerow([train_time, inf_time])
    print(f"Inference time: {inf_time:.2f}s")


if __name__ == "__main__":
    main()