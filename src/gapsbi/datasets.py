from __future__ import annotations

from typing import Protocol

import numpy as np

from gapsbi.rng import make_rngs, split_rng


class SimulatorLike(Protocol):
    def sample_theta(self, n: int, rng: np.random.Generator) -> np.ndarray: ...

    def simulate(self, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...


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
    mask = mask_generator.generate(x_full, theta, mask_rng)
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
