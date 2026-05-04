"""Inference script for generating predictions using FNO from neuraloperator."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.interpolate import interp1d

from model import FNO


# 默认路径基于脚本文件位置
_DEFAULT_CHECKPOINT_PATH = str(Path(__file__).parent / "checkpoints/fno_burgers_finetuned.pt")
_DEFAULT_TEST_DATA_PATH = str(Path(__file__).parent.parent.parent / "data/raw/task1_test.hdf5")
_DEFAULT_OUTPUT_PATH = str(Path(__file__).parent.parent.parent / "task1_pred.hdf5")


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_absolute():
        ckpt_path = get_project_root() / ckpt_path

    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 兼容两种保存格式
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        config = ckpt.get("config", {})
    else:
        state_dict = ckpt
        config = {}

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
    print("Model loaded")
    return model, config


def load_test_data(path: str, space_ds: int):
    """加载测试数据并只做空间降采样（如果需要），返回原始高分辨率和降采样后的数据。"""
    data_path = Path(path)
    if not data_path.is_absolute():
        data_path = get_project_root() / data_path

    print(f"Loading test data from {data_path}")
    with h5py.File(data_path, "r") as f:
        data = np.asarray(f["tensor"], dtype=np.float32)   # [N, 10, 1024] or [N, 10, 256]

    print(f"Original shape: {data.shape}")

    # 如果原始空间维度是 1024，则降采样到 256；否则保持
    if data.shape[-1] == 1024:
        data_ds = data[:, :, ::space_ds]   # [N, 10, 256]
    else:
        data_ds = data                     # 已经是 256
    print(f"After spatial downsampling: {data_ds.shape}")
    return data_ds, data   # 返回降采样后的输入，以及原始真实初始条件（高分辨率）


def autoregressive_predict(model, initial_condition, device, n_steps=40):
    """
    自回归预测。
    initial_condition: [N, 2, 256] 降采样后的初始条件（两个时间步）
    返回: [N, 40, 256]
    """
    n_samples, init_steps, x_grid = initial_condition.shape
    full_pred = np.zeros((n_samples, n_steps, x_grid), dtype=np.float32)
    full_pred[:, :init_steps, :] = initial_condition

    for t in range(init_steps, n_steps):
        inp = full_pred[:, t-1:t, :]  # [N, 1, X]
        inp_tensor = torch.from_numpy(inp).to(device)
        with torch.no_grad():
            full_pred[:, t:t+1, :] = model(inp_tensor).cpu().numpy()

    return full_pred


def upsample_time(data, orig_t=200):
    """时间上采样 40->200，线性插值。"""
    n_samples, n_time_down, x_grid = data.shape
    t_down = np.linspace(0, 1, n_time_down)
    t_up = np.linspace(0, 1, orig_t)
    data_2d = data.transpose(0, 2, 1).reshape(-1, n_time_down)
    f = interp1d(t_down, data_2d, kind="linear", axis=1)
    result_2d = f(t_up)
    return result_2d.reshape(n_samples, x_grid, orig_t).transpose(0, 2, 1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="FNO prediction for 1D Burgers")
    parser.add_argument("--checkpoint", type=str, default=_DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--test-data", type=str, default=_DEFAULT_TEST_DATA_PATH)
    parser.add_argument("--output", type=str, default=_DEFAULT_OUTPUT_PATH)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    model, config = load_checkpoint(args.checkpoint, device)
    space_ds = config.get("space_downsample", 4)
    time_ds = config.get("time_downsample", 5)

    # 加载测试数据（空间降采样到256，同时保留原始高分辨率用于覆盖前10步）
    test_data_ds, test_data_orig = load_test_data(args.test_data, space_ds)

    # 时间降采样，得到自回归的初始条件
    # 原始时间步为10，取::time_ds 得到索引 0,5 共2步
    initial_cond = test_data_ds[:, ::time_ds, :]   # [N, 2, 256]
    print(f"Initial condition (downsampled) shape: {initial_cond.shape}")

    print("Running autoregressive prediction...")
    inference_start = time.time()
    pred_down = autoregressive_predict(model, initial_cond, device, n_steps=40)
    inference_time = time.time() - inference_start
    print(f"Inference time: {inference_time:.2f}s")
    print(f"Prediction (downsampled): {pred_down.shape}")

    # 时间上采样到 200 步
    print("Upsampling in time...")
    pred_up = upsample_time(pred_down, orig_t=200)
    print(f"After temporal upsample: {pred_up.shape}")

    # 获取真实前10步（高分辨率）并空间降采样到256，用于覆盖
    true_ic_highres = test_data_orig[:, :10, :]           # [N, 10, 1024] or [N, 10, 256]
    if true_ic_highres.shape[-1] == 1024:
        true_ic = true_ic_highres[:, :, ::space_ds]       # [N, 10, 256]
    else:
        true_ic = true_ic_highres
    pred_up[:, :10, :] = true_ic.astype(np.float32)

    # 保存
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = get_project_root() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("data", data=pred_up)
    print(f"Saved to {out_path}")

    # 尝试读取训练时间
    train_time = 0.0
    train_time_path = Path(args.checkpoint).parent / "train_time.csv"
    if train_time_path.exists():
        with open(train_time_path, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            row = next(reader, None)
            if row:
                train_time = float(row[0])

    # 保存 task1_time.csv
    time_csv_path = out_path.parent / "task1_time.csv"
    with open(time_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["train_time", "inference_time"])
        writer.writerow([train_time, inference_time])
    print(f"Timing info saved to {time_csv_path}")


if __name__ == "__main__":
    main()