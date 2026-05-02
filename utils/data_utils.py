from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


Layout = Literal["auto", "time_x", "x_time"]


@dataclass(frozen=True)
class HDF5DatasetInfo:
    """Small metadata bundle for the selected HDF5 dataset."""

    key: str
    shape: tuple[int, ...]
    dtype: str


def _is_numeric_dataset(obj: Any) -> bool:
    return isinstance(obj, h5py.Dataset) and np.issubdtype(obj.dtype, np.number)


def _find_largest_numeric_dataset(handle: h5py.File) -> str:
    candidates: list[tuple[int, str]] = []

    def visitor(name: str, obj: Any) -> None:
        if _is_numeric_dataset(obj) and len(obj.shape) >= 3:
            candidates.append((int(np.prod(obj.shape)), name))

    handle.visititems(visitor)
    if not candidates:
        raise ValueError("No numeric HDF5 dataset with at least 3 dimensions was found.")
    candidates.sort(reverse=True)
    return candidates[0][1]


def resolve_tensor_key(handle: h5py.File, tensor_key: str | None = None) -> str:
    """Resolve the main solution tensor key for a PDEBench-style file."""

    if tensor_key is not None:
        if tensor_key not in handle:
            raise KeyError(f"HDF5 dataset key not found: {tensor_key}")
        if not _is_numeric_dataset(handle[tensor_key]):
            raise TypeError(f"HDF5 key is not a numeric dataset: {tensor_key}")
        return tensor_key

    for key in ("tensor", "data", "u", "solution"):
        if key in handle and _is_numeric_dataset(handle[key]):
            return key

    return _find_largest_numeric_dataset(handle)


class BurgersDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset for 1D Burgers HDF5 tensors.

    Returned tensors use the project convention:
    - x: [input_steps, x_grid]
    - y: [pred_steps, x_grid]

    DataLoader batches therefore become:
    - x: [batch, input_steps, x_grid]
    - y: [batch, pred_steps, x_grid]
    """

    def __init__(
        self,
        hdf5_path: str | Path,
        tensor_key: str | None = None,
        input_steps: int = 10,
        pred_steps: int = 190,
        reduced_resolution: int = 1,
        reduced_resolution_t: int = 1,
        layout: Layout = "auto",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.hdf5_path = Path(hdf5_path)
        self.input_steps = int(input_steps)
        self.pred_steps = int(pred_steps)
        self.reduced_resolution = int(reduced_resolution)
        self.reduced_resolution_t = int(reduced_resolution_t)
        self.layout = layout
        self.dtype = dtype
        self._handle: h5py.File | None = None

        if self.input_steps <= 0:
            raise ValueError("input_steps must be positive.")
        if self.pred_steps <= 0:
            raise ValueError("pred_steps must be positive.")
        if self.reduced_resolution <= 0:
            raise ValueError("reduced_resolution must be positive.")
        if self.reduced_resolution_t <= 0:
            raise ValueError("reduced_resolution_t must be positive.")
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        with h5py.File(self.hdf5_path, "r") as handle:
            self.tensor_key = resolve_tensor_key(handle, tensor_key)
            dataset = handle[self.tensor_key]
            if len(dataset.shape) < 3:
                raise ValueError(
                    "Expected selected tensor to have at least 3 dimensions: "
                    f"{self.tensor_key} shape={dataset.shape}"
                )
            self.info = HDF5DatasetInfo(
                key=self.tensor_key,
                shape=tuple(int(v) for v in dataset.shape),
                dtype=str(dataset.dtype),
            )
            self._length = int(dataset.shape[0])

            first = np.asarray(dataset[0])
            normalized = self._normalize_layout(first)
            self.sample_shape = tuple(int(v) for v in normalized.shape)
            needed_steps = self.input_steps + self.pred_steps
            if self.sample_shape[0] < needed_steps:
                raise ValueError(
                    "Not enough time steps after layout normalization and downsampling: "
                    f"have {self.sample_shape[0]}, need {needed_steps}. "
                    "Adjust input_steps, pred_steps, or reduced_resolution_t."
                )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        dataset = self._get_dataset()
        sample = np.asarray(dataset[index])
        sample = self._normalize_layout(sample)

        x_np = sample[: self.input_steps]
        y_np = sample[self.input_steps : self.input_steps + self.pred_steps]

        x = torch.as_tensor(x_np, dtype=self.dtype)
        y = torch.as_tensor(y_np, dtype=self.dtype)
        return x, y

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def _get_dataset(self) -> h5py.Dataset:
        if self._handle is None:
            self._handle = h5py.File(self.hdf5_path, "r")
        return self._handle[self.tensor_key]

    def _normalize_layout(self, sample: np.ndarray) -> np.ndarray:
        sample = np.asarray(sample)
        sample = np.squeeze(sample)
        if sample.ndim != 2:
            raise ValueError(
                "Expected one sample to reduce to a 2D [time, x] or [x, time] array, "
                f"got shape {sample.shape}."
            )

        if self.layout == "time_x":
            normalized = sample
        elif self.layout == "x_time":
            normalized = sample.T
        else:
            normalized = self._auto_layout(sample)

        normalized = normalized[:: self.reduced_resolution_t, :: self.reduced_resolution]
        return np.ascontiguousarray(normalized)

    def _auto_layout(self, sample: np.ndarray) -> np.ndarray:
        needed_steps = self.input_steps + self.pred_steps
        dim0_fits = sample.shape[0] >= needed_steps
        dim1_fits = sample.shape[1] >= needed_steps

        if dim0_fits and not dim1_fits:
            return sample
        if dim1_fits and not dim0_fits:
            return sample.T

        # For common Burgers data, time is usually the smaller axis
        # before slicing: [200, 1024], not [1024, 200].
        if sample.shape[0] <= sample.shape[1]:
            return sample
        return sample.T

    def __del__(self) -> None:
        self.close()
