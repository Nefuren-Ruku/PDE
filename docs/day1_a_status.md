# Day 1 Student A Status

Date: 2026-05-01

## Completed

- Created project directories:
  `data/raw`, `data/processed`, `src/data`, `src/models`, `tests`,
  `experiments`, `submissions`, `docs`, `configs`, `checkpoints`.
- Recorded environment in `docs/env_setup.md`.
- Implemented `src/data/burgers_dataset.py`.
- Added `tests/smoke_test_burgers_dataset.py`.
- Archived official race requirements under `docs/official/`.
- Confirmed the real Task 1 HDF5 file can be read.

## Real Data Check

File:

```text
data/raw/1D_Burgers_Sols_Nu0.001.hdf5
```

Observed datasets:

```text
t-coordinate: (202,), float32
tensor: (10000, 201, 1024), float32
x-coordinate: (1024,), float32
```

The official page describes 200 time steps, but the downloaded PDEBench tensor
contains 201 time points. Current Dataset slicing uses:

```text
input: tensor[:, 0:10, :]
target: tensor[:, 10:200, :]
```

This matches the official requirement to predict the 190 steps after the first
10 initial steps.

## Smoke Test

Command:

```powershell
python tests\smoke_test_burgers_dataset.py --hdf5-path data\raw\1D_Burgers_Sols_Nu0.001.hdf5 --reduced-resolution 4
```

Result:

```text
tensor_key: tensor
raw_shape: (10000, 201, 1024)
raw_dtype: float32
normalized_sample_shape: (201, 256)
dataset_length: 10000
input shape: (2, 10, 256)
target shape: (2, 190, 256)
input dtype: torch.float32
target dtype: torch.float32
```

## Notes

- Current Python environment is CPU-only. It is fine for Day 1 but not enough
  for meaningful FNO training.
- For Task 1 model training at official spatial resolution, pass
  `reduced_resolution=4`.
- Do not pass `reduced_resolution_t=5` for the final 190-step target shape unless
  you are intentionally reproducing a coarse-time official checkpoint workflow.
- Final submission still needs full prediction tensors shaped `[N, 200, 256]`,
  with the first 10 time steps copied from the test input exactly.
