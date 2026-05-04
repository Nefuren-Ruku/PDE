"""Prediction script for DeepONet on 1D Burgers equation."""

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

# 动态加入项目根目录
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from baselines.deeponet.model import DeepONet1D
from baselines.deeponet.data_loader import get_project_root


def load_checkpoint(ckpt_path, device, branch_dim, trunk_dim):
    """加载模型，自动适配维度"""
    model = DeepONet1D(
        branch_dim=branch_dim,
        trunk_dim=trunk_dim,
        width=50, branch_depth=2, trunk_depth=3
    ).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def upsample_time(data, orig_t=200):
    """时间上采样 40->200，线性插值"""
    n_samples, n_time_down, x_grid = data.shape
    t_down = np.linspace(0, 1, n_time_down)
    t_up = np.linspace(0, 1, orig_t)
    data_2d = data.transpose(0, 2, 1).reshape(-1, n_time_down)
    f = interp1d(t_down, data_2d, kind="linear", axis=1)
    result_2d = f(t_up)
    return result_2d.reshape(n_samples, x_grid, orig_t).transpose(0, 2, 1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="DeepONet inference")
    parser.add_argument("--checkpoint", type=str,
                        default="baselines/deeponet/checkpoints/deeponet_best.pt")
    parser.add_argument("--test-data", type=str,
                        default="data/raw/task1_test.hdf5")
    parser.add_argument("--output", type=str, default="task1_pred.hdf5",
                        help="Output HDF5 path (relative or absolute)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--time-ds", type=int, default=5)
    parser.add_argument("--space-ds", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of samples per inference batch")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    print(f"Using device: {device}")

    project_root = get_project_root()

    # 解析路径
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = project_root / ckpt_path

    test_data_path = Path(args.test_data)
    if not test_data_path.is_absolute():
        test_data_path = project_root / test_data_path

    # 读取原始测试数据（包含高分辨率空间）
    with h5py.File(test_data_path, "r") as f:
        raw_data = np.asarray(f["tensor"], dtype=np.float32)   # [N, 10, 1024] 或 [N, 10, 256]
    n_samples = raw_data.shape[0]

    # 空间降采样到256（如果原始为1024）
    if raw_data.shape[-1] == 1024:
        data_ds = raw_data[:, :, ::args.space_ds]   # [N, 10, 256]
    else:
        data_ds = np.copy(raw_data)

    # 保存真实前10步用于覆盖（空间分辨率256）
    true_ic = data_ds[:, :10, :].astype(np.float32)   # [N, 10, 256]

    # 构建降采样时间步（40步）和空间网格
    n_time_ds = 40
    n_space = data_ds.shape[-1]          # 256
    branch_dim = n_space                 # 256
    trunk_dim = 2                        # (x, t)

    # 生成时空坐标网格
    t = np.linspace(0, 1, n_time_ds)
    x = np.linspace(0, 1, n_space)
    T, X = np.meshgrid(t, x, indexing="ij")
    trunk_grid = np.stack([T.ravel(), X.ravel()], axis=1).astype(np.float32)  # [10240, 2]
    n_points = n_time_ds * n_space

    # 分支输入：取空间降采样后的第一个时间步作为初始条件
    branch_input = data_ds[:, 0, :]   # [N, 256]

    # 加载模型
    model = load_checkpoint(ckpt_path, device, branch_dim, trunk_dim)

    # 准备推理张量（所有样本放在一个Tensor中，后面按batch索引）
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

            batch_branch = branch_t[start:end]  # [b, 256]
            branch_exp = batch_branch.unsqueeze(1).expand(-1, n_points, -1)  # [b, 10240, 256]
            trunk_exp = trunk_t.unsqueeze(0).expand(b, -1, -1)  # [b, 10240, 2]
            inp = torch.cat([branch_exp, trunk_exp], dim=-1)  # [b, 10240, 258]
            inp_flat = inp.reshape(-1, branch_dim + trunk_dim)  # [b*10240, 258]

            pred_flat = model(inp_flat)  # [b*10240, 1]
            pred_batch = pred_flat.view(b, n_points).cpu().numpy()  # [b, 10240]
            pred_batch = pred_batch.reshape(b, n_time_ds, n_space)  # [b, 40, 256]
            all_pred_ds.append(pred_batch)

            if (i + 1) % 10 == 0 or i == 0 or i == n_batches - 1:
                print(f"  Batch {i + 1}/{n_batches} done")

    inf_time = time.time() - t0
    pred_ds = np.concatenate(all_pred_ds, axis=0)  # [N, 40, 256]

    # 时间上采样到 200 步
    pred_up = upsample_time(pred_ds, orig_t=200)   # [N, 200, 256]

    # 用真实前10步覆盖（确保完全一致）
    pred_up[:, :10, :] = true_ic

    # 保存预测结果
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = project_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("data", data=pred_up)
    print(f"Predictions saved to {out_path}")

    # 生成 task1_time.csv
    train_time = 0.0
    train_time_csv = ckpt_path.parent / "train_time.csv"
    if train_time_csv.exists():
        with open(train_time_csv, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            row = next(reader, None)
            if row:
                train_time = float(row[0])

    time_csv_path = out_path.parent / "task1_time.csv"
    with open(time_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["train_time", "inference_time"])
        writer.writerow([train_time, inf_time])
    print(f"Timing info saved to {time_csv_path}")
    print(f"Inference time: {inf_time:.2f}s")


if __name__ == "__main__":
    main()