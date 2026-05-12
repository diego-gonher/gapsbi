# reference code: https://github.com/Aalto-QuML/RISE/blob/main/simulators/ricker.py
from typing import Any

import numpy as np

from gapsbi.simulators.base import Simulator


class RickerSimulator(Simulator):
    """Ricker population model with Poisson observations."""

    def __init__(
        self,
        T: int = 100,
        n0: float = 1.0,
        sigma: float = 0.3,
        prior_low: np.ndarray | None = None,
        prior_high: np.ndarray | None = None,
        max_rate: float = 1e8,
        n_floor: float = 1e-12,
    ) -> None:
        self.T = T
        self.n0 = n0
        self.sigma = sigma
        self.prior_low = (
            np.array([2.0, 0.0], dtype=float)
            if prior_low is None
            else np.asarray(prior_low, dtype=float)
        )
        self.prior_high = (
            np.array([8.0, 20.0], dtype=float)
            if prior_high is None
            else np.asarray(prior_high, dtype=float)
        )
        self.max_rate = max_rate
        self.n_floor = n_floor

        if self.prior_low.shape != (2,) or self.prior_high.shape != (2,):
            raise ValueError("prior_low and prior_high must have shape (2,)")
        if self.T <= 0:
            raise ValueError("T must be positive")
        if self.n0 <= 0:
            raise ValueError("n0 must be positive")
        if self.max_rate <= 0:
            raise ValueError("max_rate must be positive")
        if self.n_floor <= 0:
            raise ValueError("n_floor must be positive")

    @property
    def name(self) -> str:
        return "ricker"

    @property
    def theta_dim(self) -> int:
        return 2

    @property
    def x_shape(self) -> tuple[int, ...]:
        return (self.T,)

    def sample_theta(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if n < 0:
            raise ValueError("n must be nonnegative")
        return rng.uniform(self.prior_low, self.prior_high, size=(n, self.theta_dim))

    def simulate(self, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        theta_array = np.asarray(theta, dtype=float)
        single = theta_array.ndim == 1

        if single:
            if theta_array.shape != (self.theta_dim,):
                raise ValueError("theta must have shape (2,) or (n, 2)")
            theta_batch = theta_array[None, :]
        elif theta_array.ndim == 2 and theta_array.shape[1] == self.theta_dim:
            theta_batch = theta_array
        else:
            raise ValueError("theta must have shape (2,) or (n, 2)")

        log_r = theta_batch[:, 0]
        phi = theta_batch[:, 1]
        if np.any(phi < 0):
            raise ValueError("phi must be nonnegative")

        n_batch = theta_batch.shape[0]
        population = np.full(n_batch, self.n0, dtype=float)
        observations = np.empty((n_batch, self.T), dtype=np.int64)
        max_exp_arg = np.log(np.finfo(float).max)

        for t in range(self.T):
            eps = rng.normal(size=n_batch)
            previous = np.maximum(population, self.n_floor)
            log_population = log_r + np.log(previous) - previous + self.sigma * eps
            population = np.exp(np.clip(log_population, None, max_exp_arg))

            rate = np.zeros(n_batch, dtype=float)
            positive_phi = phi > 0
            rate[positive_phi] = phi[positive_phi] * population[positive_phi]
            rate = np.clip(rate, 0.0, self.max_rate)
            observations[:, t] = rng.poisson(rate)

        return observations[0] if single else observations

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "theta_dim": self.theta_dim,
            "x_shape": self.x_shape,
            "T": self.T,
            "sigma": self.sigma,
            "n0": self.n0,
            "prior_low": self.prior_low.tolist(),
            "prior_high": self.prior_high.tolist(),
            "max_rate": self.max_rate,
            "n_floor": self.n_floor,
        }
