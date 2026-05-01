# Official Requirements Summary

Source of truth:

https://competition.ai4s.com.cn/race/7/description

Fetched from the official API on 2026-05-01 and archived under
`docs/official/`.

## Race

- Race: AI4S智能体CNS挑战赛——神经算子PDE智能体
- Stage: 初赛
- Current official status: IN_PROGRESS
- Initial round season window shown by API: 2026-02-28 10:00:00 to
  2026-05-25 14:00:00

## Core Task

Build a PDE research workflow Agent for neural operators. The competition
evaluates the Agent research loop, not only a trained model checkpoint:

- understand PDEBench and neural-operator baselines;
- diagnose bottlenecks from logs or scientific data;
- modify model code, loss functions, or physics constraints;
- run experiments, debug failures, and keep scientific logs.

## Data

Task 1:

- Fixed physical environment, Burgers equation with `Nu=0.001`.
- Official PDEBench data file: `1D_Burgers_Sols_Nu0.001.hdf5`.
- Official data shape: 10000 samples, 200 time steps, 1024 spatial points.
- Official checkpoints allowed for Task 1 fine-tuning:
  `1D_Burgers_Sols_Nu0.001_FNO.pt` and
  `1D_Burgers_Sols_Nu0.001_Unet-PF-20.pt`.
- Official checkpoint training used `reduced_resolution_t=5` and
  `reduced_resolution=4`, so model-level validation is commonly 40 time steps
  and 256 spatial points.

Task 2:

- Multi-physics Burgers prediction across multiple viscosity coefficients.
- Training data provides each sample's `Nu`.
- Test data does not provide `Nu`; inference must use initial conditions only.
- Task 2 must be trained from scratch and must not use Task 1 data or Task 1
  checkpoints.
- No extra numerical-solver-generated training data is allowed.

## Prediction Shape

For both tasks, the test input gives the first 10 time steps. The model must
predict future steps, but the submitted HDF5 prediction file must contain the
full 200-step tensor:

```text
task{N}_pred.hdf5
tensor: [N, 200, 256]
```

The first 10 time steps must be copied from the test input and match ground
truth within tolerance `1e-3`. Time steps 10-199 are the 190 prediction steps.

## Submission Package

Submit `submission.zip` containing:

```text
submission/
├── submission.json
├── task1_pred.hdf5
├── task1_time.csv
├── task1_logs.log
├── task2_pred.hdf5
├── task2_time.csv
├── task2_logs.log
├── methodology.pdf
└── code/
```

At least one task may be submitted. For each submitted task, all three task files
are required:

```text
task{N}_pred.hdf5
task{N}_time.csv
task{N}_logs.log
```

`submission.json`:

```json
{
  "submission_id": "你的队伍名称",
  "problem_id": "PDE_Burgers",
  "code_path": "code"
}
```

`task{N}_time.csv` must include:

```text
train_time,inference_time
```

Units are seconds. `train_time` includes Agent thinking/reasoning time.

`task{N}_logs.log` must include Agent trace and experiment tracking. The official
FAQ says logs should record wall-clock time for Agent thinking and model
training so `time.csv` can be checked.

## Scoring

Total score is Task 1 + Task 2, maximum 300.

Task 1 maximum 150:

- prediction accuracy score: segmented prediction score times 0.75;
- training-time score: official page lists thresholds for Agent + model training
  time;
- inference-time score: 0-2 minutes linearly decays, and over 2 minutes gives
  zero for the task.

Task 2 maximum 150:

- segmented prediction score times 1.5;
- training time is not scored, but total duration must be within 12 hours;
- inference over 2 minutes gives zero for the task.

Segmented prediction score:

- scoring uses only the 190 prediction steps after the first 10 initial steps;
- segment 1, steps 0-47, weight 25%:
  `100 * exp(-20 * Rel-MSE)`;
- segment 2, steps 47-95, weight 25%:
  `100 * exp(-10 * Rel-MSE)`;
- segment 3, steps 95-190, weight 50%:
  max of Lorentzian and Frechet components.

Rel-MSE is computed per sample and per time step:

```text
rel_t = sum((pred - gt)^2) / sum(gt^2)
```

Then average over time steps, cap each sample at 5.0, and average over samples.

## Notes To Recheck Before Submission

- The official description currently contains an apparent inconsistency in Task
  1 timing subscore maxima versus listed point values. Use the live official
  page and sample package as the final authority when packaging.
- The sample submission zip must be inspected before final export.
- Do not publish official private datasets, checkpoints, or generated outputs
  outside the competition workflow.
