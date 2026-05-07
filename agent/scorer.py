"""Local scoring helpers for sequence predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.fft import rfft


def _safe_rel_mse(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute capped relative MSE averaged per sample and time step."""
    diff_sq = np.sum((pred - gt) ** 2, axis=-1)
    gt_sq = np.sum(gt**2, axis=-1)
    rel = diff_sq / np.maximum(gt_sq, 1.0e-12)
    per_sample = np.mean(rel, axis=1)
    per_sample = np.minimum(per_sample, 5.0)
    return float(np.mean(per_sample))


def _segment_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute RMSE over all values in a segment."""
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def _fourier_distance(pred: np.ndarray, gt: np.ndarray, top_k: int = 10) -> float:
    """Compute a relative top-k Fourier magnitude distance."""
    pred_fft = np.abs(rfft(pred, axis=-1))
    gt_fft = np.abs(rfft(gt, axis=-1))
    k = min(top_k, pred_fft.shape[-1])
    gt_indices = np.argsort(gt_fft, axis=-1)[..., -k:]
    pred_top = np.take_along_axis(pred_fft, gt_indices, axis=-1)
    gt_top = np.take_along_axis(gt_fft, gt_indices, axis=-1)
    numerator = np.linalg.norm(pred_top - gt_top, axis=-1)
    denominator = np.maximum(np.linalg.norm(gt_top, axis=-1), 1.0e-12)
    return float(np.mean(numerator / denominator))


def compute_segment_score(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Compute three-segment score for prediction arrays.

    Args:
        pred: Prediction array with shape ``[n_samples, 190, 256]``.
        gt: Ground-truth array with shape ``[n_samples, 190, 256]``.

    Returns:
        Segment scores and weighted final score.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction and ground truth shapes differ: {pred.shape} vs {gt.shape}")
    if pred.ndim != 3 or pred.shape[1] != 190:
        raise ValueError(f"Expected shape [n_samples, 190, n_x], got {pred.shape}")

    seg1_pred, seg1_gt = pred[:, 0:47, :], gt[:, 0:47, :]
    seg2_pred, seg2_gt = pred[:, 47:95, :], gt[:, 47:95, :]
    seg3_pred, seg3_gt = pred[:, 95:190, :], gt[:, 95:190, :]

    rel1 = _safe_rel_mse(seg1_pred, seg1_gt)
    rel2 = _safe_rel_mse(seg2_pred, seg2_gt)
    score1 = float(100.0 * np.exp(-20.0 * rel1))
    score2 = float(100.0 * np.exp(-10.0 * rel2))

    rmse3 = _segment_rmse(seg3_pred, seg3_gt)
    fd3 = _fourier_distance(seg3_pred, seg3_gt)
    lorentzian = 100.0 / (1.0 + 10.0 * rmse3)
    frechet = 50.0 * np.exp(-(fd3**2))
    score3 = float(max(lorentzian, frechet))

    final_score = float(0.25 * score1 + 0.25 * score2 + 0.5 * score3)
    return {
        "score1": score1,
        "score2": score2,
        "score3": score3,
        "final_score": final_score,
    }


def _load_hdf5_array(path: Path) -> np.ndarray:
    """Load a prediction or ground-truth array from an HDF5 file."""
    with h5py.File(path, "r") as handle:
        if "data" in handle:
            key = "data"
        elif "tensor" in handle:
            key = "tensor"
        else:
            dataset_keys = [name for name, value in handle.items() if isinstance(value, h5py.Dataset)]
            if not dataset_keys:
                raise ValueError(f"No datasets found in {path}")
            key = dataset_keys[0]
        return np.asarray(handle[key])


def _strip_initial_steps(array: np.ndarray) -> np.ndarray:
    """Return the 190-step evaluation window."""
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {array.shape}")
    if array.shape[1] >= 200:
        return array[:, 10:200, :]
    if array.shape[1] == 190:
        return array
    raise ValueError(f"Expected 190 or at least 200 time steps, got shape {array.shape}")


def evaluate_submission(pred_hdf5_path: str, gt_hdf5_path: str) -> dict[str, float]:
    """Load HDF5 arrays and compute the local segment score."""
    pred = _strip_initial_steps(_load_hdf5_array(Path(pred_hdf5_path)))
    gt = _strip_initial_steps(_load_hdf5_array(Path(gt_hdf5_path)))
    return compute_segment_score(pred, gt)
