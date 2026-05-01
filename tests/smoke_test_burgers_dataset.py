from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.burgers_dataset import BurgersDataset


def create_synthetic_hdf5(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n_samples, n_time, n_x = 8, 200, 256
    x = np.linspace(0.0, 1.0, n_x, dtype=np.float32)
    t = np.linspace(0.0, 1.0, n_time, dtype=np.float32)

    tensor = np.empty((n_samples, n_time, n_x), dtype=np.float32)
    for i in range(n_samples):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        speed = rng.uniform(0.5, 1.5)
        decay = rng.uniform(0.2, 0.8)
        wave = np.sin(2.0 * np.pi * (x[None, :] - speed * t[:, None]) + phase)
        tensor[i] = wave * np.exp(-decay * t[:, None])

    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=tensor)
        handle.create_dataset("x-coordinate", data=x)
        handle.create_dataset("t-coordinate", data=t)
        handle.create_dataset("nu", data=np.full((n_samples,), 0.001, dtype=np.float32))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for BurgersDataset.")
    parser.add_argument(
        "--hdf5-path",
        type=Path,
        default=PROJECT_ROOT / "tests" / "fixtures" / "synthetic_burgers.hdf5",
        help="Path to an official or synthetic Burgers HDF5 file.",
    )
    parser.add_argument("--tensor-key", default=None)
    parser.add_argument("--input-steps", type=int, default=10)
    parser.add_argument("--pred-steps", type=int, default=190)
    parser.add_argument("--reduced-resolution", type=int, default=1)
    parser.add_argument("--reduced-resolution-t", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.hdf5_path.exists():
        print(f"Creating synthetic HDF5 fixture: {args.hdf5_path}")
        create_synthetic_hdf5(args.hdf5_path)

    dataset = BurgersDataset(
        hdf5_path=args.hdf5_path,
        tensor_key=args.tensor_key,
        input_steps=args.input_steps,
        pred_steps=args.pred_steps,
        reduced_resolution=args.reduced_resolution,
        reduced_resolution_t=args.reduced_resolution_t,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    x, y = next(iter(loader))

    print(f"hdf5_path: {args.hdf5_path}")
    print(f"tensor_key: {dataset.info.key}")
    print(f"raw_shape: {dataset.info.shape}")
    print(f"raw_dtype: {dataset.info.dtype}")
    print(f"normalized_sample_shape: {dataset.sample_shape}")
    print(f"dataset_length: {len(dataset)}")
    print(f"input shape: {tuple(x.shape)}")
    print(f"target shape: {tuple(y.shape)}")
    print(f"input dtype: {x.dtype}")
    print(f"target dtype: {y.dtype}")
    print(f"input min/max: {float(x.min()):.6f}/{float(x.max()):.6f}")
    print(f"target min/max: {float(y.min()):.6f}/{float(y.max()):.6f}")
    dataset.close()


if __name__ == "__main__":
    main()
