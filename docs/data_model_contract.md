# Data Model Contract

This document fixes the Day 1 data shape convention for the Burgers task.
The source of truth is the official race page:
https://competition.ai4s.com.cn/race/7/description

## Raw HDF5

Expected official file:

```text
data/raw/1D_Burgers_Sols_Nu0.001.hdf5
```

The Dataset first looks for a dataset named `tensor`. If it is not present, it
falls back to the largest numeric dataset in the HDF5 file. The real key and
shape must be confirmed again with the official sample submission.

Official Task 1 training data:

```text
10000 samples, 200 time steps, 1024 spatial points
```

Local file check on 2026-05-01:

```text
data/raw/1D_Burgers_Sols_Nu0.001.hdf5
tensor: (10000, 201, 1024), float32
t-coordinate: (202,), float32
x-coordinate: (1024,), float32
```

The local tensor has 201 time points. For the current competition convention, we
slice the first 10 time steps as input and the next 190 time steps as target,
using 200 time steps total and leaving the extra final point unused for Day 1
training targets.

Official test/submission resolution:

```text
N samples, 200 time steps, 256 spatial points
```

## Internal Sample Layout

Each raw sample is normalized to:

```text
[time, x]
```

The Dataset supports common raw sample layouts:

- `[time, x]`
- `[x, time]`
- layouts with singleton dimensions, such as `[time, x, 1]`

## Model Input

For `input_steps = 10`:

```text
x: [batch, 10, x_grid]
```

## Model Target

For `pred_steps = 190`:

```text
y: [batch, 190, x_grid]
```

The target starts immediately after the input window, so it corresponds to the
190 prediction steps after the first 10 conditioning steps.

This is the training/evaluation target only. It is not the final submission
shape.

## Downsampling Parameters

The Dataset reserves:

- `reduced_resolution`: spatial stride
- `reduced_resolution_t`: temporal stride

Both are applied before slicing input and target windows:

```text
sample = sample[::reduced_resolution_t, ::reduced_resolution]
```

If temporal downsampling makes the sample shorter than
`input_steps + pred_steps`, reduce `pred_steps` for experiments or do not use
temporal downsampling for submission-shaped training.

For the current official Task 1 HDF5 file, use `reduced_resolution=4` and
`reduced_resolution_t=1` to get:

```text
normalized sample: [201, 256]
model input: [batch, 10, 256]
model target: [batch, 190, 256]
```

## Submission Tensor

Official submission files `task1_pred.hdf5` and `task2_pred.hdf5` must contain a
full tensor with:

```text
tensor: [N, 200, 256]
```

The first 10 time steps are the initial condition and must match the official
test input exactly within tolerance `1e-3`. Only time steps 10-199 are model
predictions.

## Task 2 Constraint

Task 2 training data provides `Nu`, but test data does not. Training code may use
`Nu`, but inference must be possible from initial conditions alone. Task 2 must
not reuse Task 1 data or Task 1 checkpoints.
