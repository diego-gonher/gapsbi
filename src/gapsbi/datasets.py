from __future__ import annotations

from typing import Protocol

import numpy as np
from tqdm.auto import tqdm

from gapsbi.rng import make_rngs, split_rng


class SimulatorLike(Protocol):
    def sample_theta(self, n: int, rng: np.random.Generator) -> np.ndarray: ...

    def simulate(self, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...

    def metadata(self) -> dict: ...


class MaskGeneratorLike(Protocol):
    def generate(
        self,
        x_full: np.ndarray,
        theta: np.ndarray | None,
        rng: np.random.Generator,
    ) -> np.ndarray: ...


def generate_split(
    simulator: SimulatorLike,
    mask_generator: MaskGeneratorLike,
    n: int,
    rng: np.random.Generator,
    progress: bool = False,
    desc: str = "Generating split",
) -> dict[str, np.ndarray]:
    """Generate one GAPSBI dataset split."""
    if n < 0:
        raise ValueError("n must be nonnegative.")

    theta_rng, sim_rng, mask_rng = split_rng(rng, 3)
    theta = simulator.sample_theta(n, theta_rng)
    x_full_items = []
    mask_items = []

    iterator = tqdm(
        range(n),
        desc=desc,
        unit="sim",
        disable=not progress,
        dynamic_ncols=True,
        mininterval=0.1,
        smoothing=0.0,
    )
    for i in iterator:
        x_i = simulator.simulate(theta[i], sim_rng)
        mask_i = _generate_mask(simulator, mask_generator, x_i, theta[i], mask_rng)
        x_full_items.append(x_i)
        mask_items.append(mask_i)

    x_full = np.stack(x_full_items) if x_full_items else np.empty((0, *simulator.x_shape))
    mask = np.stack(mask_items) if mask_items else np.empty_like(x_full, dtype=np.int8)
    x_obs = x_full * mask

    return {
        "theta": theta,
        "x_full": x_full,
        "x_obs": x_obs,
        "mask": mask,
    }


def generate_dataset(
    simulator: SimulatorLike,
    mask_generator: MaskGeneratorLike,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    progress: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    """Generate independent train, validation, and test splits."""
    split_rngs = make_rngs(seed, ["train", "val", "test"])

    return {
        "train": generate_split(
            simulator,
            mask_generator,
            n_train,
            split_rngs["train"],
            desc="Generating train",
            progress=progress,
        ),
        "val": generate_split(
            simulator,
            mask_generator,
            n_val,
            split_rngs["val"],
            desc="Generating val",
            progress=progress,
        ),
        "test": generate_split(
            simulator,
            mask_generator,
            n_test,
            split_rngs["test"],
            desc="Generating test",
            progress=progress,
        ),
    }


def apply_train_val_sample_limits(
    dataset: dict[str, dict[str, np.ndarray]],
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Optionally slice train/validation splits while preserving the full test split."""
    if max_train_samples is None and max_val_samples is None:
        return dataset

    max_train_samples = _validate_sample_limit(max_train_samples, "max_train_samples")
    max_val_samples = _validate_sample_limit(max_val_samples, "max_val_samples")

    limited = {
        split: dict(split_data)
        for split, split_data in dataset.items()
    }
    if max_train_samples is not None:
        limited["train"] = _slice_split(limited["train"], max_train_samples)
    if max_val_samples is not None:
        limited["val"] = _slice_split(limited["val"], max_val_samples)
    return limited


def _validate_sample_limit(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be positive when set, got {value}.")
    return value


def _slice_split(split_data: dict[str, np.ndarray], max_samples: int) -> dict[str, np.ndarray]:
    return {
        name: array[:max_samples]
        for name, array in split_data.items()
    }


def _generate_mask(
    simulator: SimulatorLike,
    mask_generator: MaskGeneratorLike,
    x_full: np.ndarray,
    theta: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    metadata = simulator.metadata()
    if metadata.get("name") != "spatial_sir":
        return mask_generator.generate(x_full, theta, rng)
    return _generate_spatial_sir_mask(metadata, mask_generator, x_full, theta, rng)


def _generate_spatial_sir_mask(
    simulator_metadata: dict,
    mask_generator: MaskGeneratorLike,
    x_full: np.ndarray,
    theta: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    original_x_shape = tuple(simulator_metadata["original_x_shape"])
    flatten = bool(simulator_metadata.get("flatten", True))
    channels, height, width = original_x_shape
    single = (flatten and x_full.ndim == 1) or (not flatten and x_full.ndim == 3)

    if single:
        snapshots = x_full.reshape((1, channels, height, width))
        theta_batch = np.asarray(theta)[None, ...]
    else:
        n = x_full.shape[0]
        snapshots = x_full.reshape((n, channels, height, width)) if flatten else x_full
        theta_batch = theta

    # Encode one scalar state per cell for value-dependent masks; shape-only masks ignore values.
    n_snapshots = snapshots.shape[0]
    cell_values = np.argmax(snapshots, axis=1).reshape((n_snapshots, height * width))
    cell_mask = mask_generator.generate(cell_values, theta_batch, rng).reshape(
        (n_snapshots, height, width)
    )
    expanded_mask = np.broadcast_to(
        cell_mask[:, None, :, :],
        (n_snapshots, channels, height, width),
    )

    if single:
        return expanded_mask[0].reshape(x_full.shape).astype(np.int8)
    if flatten:
        return expanded_mask.reshape(x_full.shape).astype(np.int8)
    return expanded_mask.astype(np.int8)
