from __future__ import annotations

from typing import Protocol

import numpy as np

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
) -> dict[str, np.ndarray]:
    """Generate one GAPSBI dataset split."""
    if n < 0:
        raise ValueError("n must be nonnegative.")

    theta_rng, sim_rng, mask_rng = split_rng(rng, 3)
    theta = simulator.sample_theta(n, theta_rng)
    x_full = simulator.simulate(theta, sim_rng)
    mask = _generate_mask(simulator, mask_generator, x_full, theta, mask_rng)
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
) -> dict[str, dict[str, np.ndarray]]:
    """Generate independent train, validation, and test splits."""
    split_rngs = make_rngs(seed, ["train", "val", "test"])

    return {
        "train": generate_split(simulator, mask_generator, n_train, split_rngs["train"]),
        "val": generate_split(simulator, mask_generator, n_val, split_rngs["val"]),
        "test": generate_split(simulator, mask_generator, n_test, split_rngs["test"]),
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
    n = x_full.shape[0]

    snapshots = x_full.reshape((n, channels, height, width)) if flatten else x_full
    # Encode one scalar state per cell for value-dependent masks; shape-only masks ignore values.
    cell_values = np.argmax(snapshots, axis=1).reshape((n, height * width))
    cell_mask = mask_generator.generate(cell_values, theta, rng).reshape((n, height, width))
    expanded_mask = np.broadcast_to(cell_mask[:, None, :, :], (n, channels, height, width))

    if flatten:
        return expanded_mask.reshape(x_full.shape).astype(np.int8)
    return expanded_mask.astype(np.int8)
