from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np


SPLITS = ("train", "val", "test")
ARRAYS = ("theta", "x_full", "x_obs", "mask")


def save_gapsbi_hdf5(
    path: str | os.PathLike[str],
    dataset: dict[str, dict[str, np.ndarray]],
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    """Save a GapSBI dataset using the grouped HDF5 contract."""
    validate_gapsbi_dataset(dataset)

    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5:
        for split in SPLITS:
            group = h5.create_group(split)
            for name in ARRAYS:
                group.create_dataset(name, data=dataset[split][name])

        if metadata is not None:
            for key, value in metadata.items():
                h5.attrs[key] = _metadata_attr_value(value)


def load_gapsbi_hdf5(
    path: str | os.PathLike[str],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Load a GapSBI HDF5 dataset and file-level metadata."""
    dataset: dict[str, dict[str, np.ndarray]] = {}
    metadata: dict[str, Any] = {}

    with h5py.File(path, "r") as h5:
        for split in SPLITS:
            dataset[split] = {name: h5[split][name][...] for name in ARRAYS}
        metadata = {key: _decode_metadata_attr(value) for key, value in h5.attrs.items()}

    return dataset, metadata


def validate_gapsbi_dataset(dataset: dict[str, dict[str, np.ndarray]]) -> None:
    """Validate the GapSBI in-memory dataset contract."""
    for split in SPLITS:
        if split not in dataset:
            raise ValueError(f"Missing required split: {split}.")

        split_data = dataset[split]
        for name in ARRAYS:
            if name not in split_data:
                raise ValueError(f"Missing required array for {split}: {name}.")

        theta = np.asarray(split_data["theta"])
        x_full = np.asarray(split_data["x_full"])
        x_obs = np.asarray(split_data["x_obs"])
        mask = np.asarray(split_data["mask"])

        if theta.ndim != 2:
            raise ValueError(f"{split}/theta must be 2D.")
        if x_full.shape != x_obs.shape or x_full.shape != mask.shape:
            raise ValueError(f"{split}/x_full, x_obs, and mask must have matching shapes.")
        if x_full.shape[0] != theta.shape[0]:
            raise ValueError(f"{split}/x_full rows must match theta rows.")

        for name, array in {
            "theta": theta,
            "x_full": x_full,
            "x_obs": x_obs,
            "mask": mask,
        }.items():
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{split}/{name} must contain only finite values.")

        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError(f"{split}/mask must be binary.")
        if not np.array_equal(x_obs, x_full * mask):
            raise ValueError(f"{split}/x_obs must equal x_full * mask.")


def _metadata_attr_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str | int | float | bool):
        return value
    return json.dumps(value)


def _decode_metadata_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(value, np.generic):
        return value.item()
    return value
