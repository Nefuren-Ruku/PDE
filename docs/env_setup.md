# Day 1 Environment Setup

This file records the local environment used for Day 1, Student A tasks.

## Python

- Python executable: `D:\python解释器\python.exe`
- Python version: `3.12.5`

## Hardware

- CUDA available: `False`
- CUDA version: `None`
- GPU count: `0`

The current interpreter is CPU-only. This is enough for Day 1 data loading and
smoke tests, but real FNO training will need a CUDA-enabled PyTorch environment
or a remote GPU.

## Installed Packages

Verified on 2026-05-01:

| Package | Version |
| --- | --- |
| torch | 2.10.0+cpu |
| h5py | 3.14.0 |
| numpy | 2.3.1 |
| matplotlib | 3.10.9 |
| scipy | 1.17.1 |
| pyyaml | 6.0.3 |
| tqdm | 4.67.1 |
| neuraloperator / neuralop | 2.0.0 |

## Install Command

If this environment needs to be recreated, run:

```powershell
python -m pip install torch h5py numpy matplotlib scipy pyyaml tqdm neuraloperator
```

For GPU training, install a CUDA build of PyTorch from the official PyTorch
selector and then reinstall the remaining packages.

## Project Directories

Created for the competition workflow:

```text
data/raw/
data/processed/
src/data/
src/models/
tests/
experiments/
submissions/
docs/
configs/
checkpoints/
```

## Day 1 Status (Updated 2026-05-01)

- The Python environment can import all required Day 1 packages.
- GPU is not available in the current interpreter.
- `src/data/burgers_dataset.py` implemented with HDF5 auto-key detection, auto-layout recognition, and spatial/temporal downsampling support.
- Real HDF5 data file verified:
  - Path: `data/raw/1D_Burgers_Sols_Nu0.001.hdf5`
  - Shape: `tensor (10000, 201, 1024)`, float32 — 201 time steps (1 more than the official description of 200).
  - Slicing: input `0:10`, target `10:200` → 190 prediction steps, matching the official requirement.
- Smoke test passed with real data:
  ```
  python tests\smoke_test_burgers_dataset.py --hdf5-path data\raw\1D_Burgers_Sols_Nu0.001.hdf5 --reduced-resolution 4
  ```
  Output: input shape `(2, 10, 256)`, target shape `(2, 190, 256)`, dtype `torch.float32`.
- Official competition requirements archived under `docs/official/`.
