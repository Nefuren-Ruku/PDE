"""Training script for PI-DeepONet on 1D Burgers equation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 修正导入路径
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from baselines.pi_deeponet.model import PIDeepONet1D
from baselines.deeponet.data_loader import load_deeponet_data  # 复用相同的数据加载


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PI-DeepONet for 1D Burgers")
    parser.add_argument("--train-data", type=str, default="data/raw/1D_Burgers_Sols_Nu0.001.hdf5")
    parser.add_argument("--val-data", type=str, default="data/raw/task1_val.hdf5")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Very small batch size due to high memory cost of double backprop")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--time-ds", type=int, default=5)
    parser.add_argument("--space-ds", type=int, default=4)
    parser.add_argument("--lambda-data", type=float, default=1.0,
                        help="Weight of data loss")
    parser.add_argument("--lambda-physics", type=float, default=0.1,
                        help="Weight of physics (PDE residual) loss")
    parser.add_argument("--nu", type=float, default=0.001,
                        help="Viscosity coefficient of Burgers equation")
    parser.add_argument("--checkpoint-dir", type=str, default="baselines/pi_deeponet/checkpoints")
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    # 1. 加载数据（与 DeepONet 完全相同）
    train_dataset, val_dataset = load_deeponet_data(
        train_path=args.train_data,
        val_path=args.val_data,
        time_ds=args.time_ds,
        space_ds=args.space_ds,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) if val_dataset else None

    # 2. 创建模型
    model = PIDeepONet1D(nu=args.nu).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 3. 优化器、调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion_data = nn.MSELoss()

    # 4. 训练循环
    best_val_loss = float("inf")
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_data_loss = 0.0
        total_physics_loss = 0.0

        for branch, trunk, target in train_loader:
            branch = branch.to(device)          # [B, 256]
            trunk = trunk.to(device)            # [B, 10240, 2]
            target = target.to(device)          # [B, 10240]

            B, N, _ = trunk.shape

            # ---- 数据损失 ----
            branch_exp = branch.unsqueeze(1).expand(-1, N, -1)       # [B, 10240, 256]
            inputs = torch.cat([branch_exp, trunk], dim=-1)          # [B, 10240, 258]
            inputs_flat = inputs.reshape(-1, 258)
            pred_flat = model(inputs_flat)                           # [B*10240, 1]
            pred = pred_flat.view(B, N)                              # [B, 10240]
            loss_data = criterion_data(pred, target)

            # ---- 物理损失（PDE 残差） ----
            residual = model.compute_pde_residual(branch, trunk)     # [B, 10240]
            loss_physics = torch.mean(residual ** 2)

            # 总损失
            loss = args.lambda_data * loss_data + args.lambda_physics * loss_physics

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_data_loss += loss_data.item() * B
            total_physics_loss += loss_physics.item() * B

        avg_data_loss = total_data_loss / len(train_dataset)
        avg_physics_loss = total_physics_loss / len(train_dataset)
        scheduler.step()

        # 验证（仅计算数据损失，物理损失供参考）
        if val_loader:
            model.eval()
            val_data_loss = 0.0
            with torch.no_grad():
                for branch, trunk, target in val_loader:
                    branch = branch.to(device)
                    trunk = trunk.to(device)
                    target = target.to(device)
                    B, N, _ = trunk.shape
                    branch_exp = branch.unsqueeze(1).expand(-1, N, -1)
                    inputs = torch.cat([branch_exp, trunk], dim=-1)
                    inputs_flat = inputs.reshape(-1, 258)
                    pred_flat = model(inputs_flat)
                    pred = pred_flat.view(B, N)
                    val_data_loss += criterion_data(pred, target).item() * B
            avg_val_loss = val_data_loss / len(val_dataset)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), checkpoint_dir / "pi_deeponet_best.pt")
        else:
            avg_val_loss = float("nan")

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{args.epochs} | "
                  f"Data Loss: {avg_data_loss:.6e} | Physics Loss: {avg_physics_loss:.6e} | "
                  f"Val Loss: {avg_val_loss:.6e} | LR: {scheduler.get_last_lr()[0]:.2e}")

    train_time = time.time() - start_time
    print(f"Training finished. Total time: {train_time:.2f}s")

    # 保存最终模型
    torch.save(model.state_dict(), checkpoint_dir / "pi_deeponet_final.pt")
    print(f"Final model saved to {checkpoint_dir / 'pi_deeponet_final.pt'}")

    # 记录训练时间
    time_csv_path = checkpoint_dir / "task1_time.csv"
    with open(time_csv_path, "w") as f:
        f.write("train_time,inference_time\n")
        f.write(f"{train_time:.2f},0.0\n")


if __name__ == "__main__":
    main()