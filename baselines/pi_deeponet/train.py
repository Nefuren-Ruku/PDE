"""Training script for PI-DeepONet on 1D Burgers equation."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.pi_deeponet.model import PIDeepONet1D
from baselines.deeponet.data_loader import load_deeponet_data


def parse_args():
    parser = argparse.ArgumentParser(description="Train PI-DeepONet for 1D Burgers")
    parser.add_argument("--train-data", default=str(PROJECT_ROOT / "data/raw/1D_Burgers_Sols_Nu0.001.hdf5"))
    parser.add_argument("--val-data", default=str(PROJECT_ROOT / "data/raw/task1_val.hdf5"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)  # 必须很小，因为需要高阶梯度
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--time-ds", type=int, default=5)
    parser.add_argument("--space-ds", type=int, default=4)
    parser.add_argument("--lambda-data", type=float, default=1.0)
    parser.add_argument("--lambda-physics", type=float, default=0.1)
    parser.add_argument("--nu", type=float, default=0.001)
    parser.add_argument("--checkpoint-dir", default=str(PROJECT_ROOT / "baselines/pi_deeponet/checkpoints"))
    return parser.parse_args()


def get_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    # 数据加载
    train_ds, val_ds = load_deeponet_data(
        args.train_data, args.val_data, args.time_ds, args.space_ds
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size) if val_ds else None

    # 推断维度（与 DeepONet 相同）
    sample_branch, sample_trunk, _ = train_ds[0]
    branch_dim = sample_branch.shape[0]      # 256
    trunk_dim = sample_trunk.shape[-1]       # 2
    n_points = sample_trunk.shape[0]         # 10240

    model = PIDeepONet1D(nu=args.nu).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion_data = nn.MSELoss()

    best_val = float("inf")
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_data_loss = 0.0
        total_phys_loss = 0.0
        for branch, trunk, target in train_loader:
            branch = branch.to(device)          # [B, 256]
            trunk = trunk.to(device)            # [B, 10240, 2]
            target = target.to(device)          # [B, 10240]

            B, N, _ = trunk.shape

            # 数据损失（展平输入）
            branch_exp = branch.unsqueeze(1).expand(-1, N, -1)        # [B, N, 256]
            inp = torch.cat([branch_exp, trunk], dim=-1)              # [B, N, 258]
            inp_flat = inp.reshape(-1, branch_dim + trunk_dim)        # 动态维度
            pred_flat = model(inp_flat)                               # [B*N]
            pred = pred_flat.view(B, N)
            loss_data = criterion_data(pred, target)

            # 物理损失
            residual = model.compute_pde_residual(branch, trunk)      # [B, N]
            loss_physics = torch.mean(residual ** 2)

            loss = args.lambda_data * loss_data + args.lambda_physics * loss_physics

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_data_loss += loss_data.item() * B
            total_phys_loss += loss_physics.item() * B

        avg_train_data = total_data_loss / len(train_ds)
        avg_train_phys = total_phys_loss / len(train_ds)
        scheduler.step()

        # 验证
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for branch, trunk, target in val_loader:
                    branch = branch.to(device)
                    trunk = trunk.to(device)
                    target = target.to(device)
                    B, N, _ = trunk.shape
                    branch_exp = branch.unsqueeze(1).expand(-1, N, -1)
                    inp = torch.cat([branch_exp, trunk], dim=-1)
                    inp_flat = inp.reshape(-1, branch_dim + trunk_dim)
                    pred_flat = model(inp_flat)
                    pred = pred_flat.view(B, N)
                    val_loss += criterion_data(pred, target).item() * B
            avg_val = val_loss / len(val_ds)
            if avg_val < best_val:
                best_val = avg_val
                torch.save(model.state_dict(), ckpt_dir / "pi_deeponet_best.pt")
        else:
            avg_val = float("nan")

        if epoch % 10 == 1 or epoch == 1:
            print(f"Epoch {epoch}/{args.epochs}  Data {avg_train_data:.6e}  Phys {avg_train_phys:.6e}  Val {avg_val:.6e}  LR {scheduler.get_last_lr()[0]:.2e}")

    train_time = time.time() - t0
    print(f"Training time: {train_time:.2f}s")

    torch.save(model.state_dict(), ckpt_dir / "pi_deeponet_final.pt")
    with open(ckpt_dir / "train_time.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["train_time"])
        w.writerow([train_time])
    print("Done.")


if __name__ == "__main__":
    main()