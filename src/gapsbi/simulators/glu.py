# reference code: https://github.com/sbi-benchmark/sbibm/blob/main/sbibm/tasks/gaussian_linear_uniform/task.py
from typing import Any

import numpy as np

from gapsbi.simulators.base import Simulator


class GLUSimulator(Simulator):
    """Gaussian Linear Uniform simulator."""

    def __init__(
        self,
        dim: int = 10,
        prior_bound: float = 1.0,
        simulator_scale: float = 0.1,
    ) -> None:
        self.dim = int(dim)
        self.prior_bound = float(prior_bound)
        self.simulator_scale = float(simulator_scale)

        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.prior_bound <= 0:
            raise ValueError("prior_bound must be positive")
        if self.simulator_scale < 0:
            raise ValueError("simulator_scale must be nonnegative")

        self.prior_low = np.full(self.dim, -self.prior_bound, dtype=float)
        self.prior_high = np.full(self.dim, self.prior_bound, dtype=float)

    @property
    def name(self) -> str:
        return "glu"

    @property
    def theta_dim(self) -> int:
        return self.dim

    @property
    def x_shape(self) -> tuple[int, ...]:
        return (self.dim,)

    def sample_theta(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if n < 0:
            raise ValueError("n must be nonnegative")
        return rng.uniform(self.prior_low, self.prior_high, size=(n, self.theta_dim))

    def simulate(self, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        theta_array = np.asarray(theta, dtype=float)
        single = theta_array.ndim == 1

        if theta_array.ndim < 1 or theta_array.shape[-1] != self.theta_dim:
            raise ValueError(f"theta must have shape ({self.theta_dim},) or (batch, {self.theta_dim})")
        if single:
            theta_batch = theta_array[None, :]
        elif theta_array.ndim == 2:
            theta_batch = theta_array
        else:
            raise ValueError(f"theta must have shape ({self.theta_dim},) or (batch, {self.theta_dim})")

        noise = rng.normal(loc=0.0, scale=self.simulator_scale, size=theta_batch.shape)
        x = theta_batch + noise

        if not np.all(np.isfinite(x)):
            raise FloatingPointError("GLU simulation produced nonfinite values")

        return x[0] if single else x

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "theta_dim": self.theta_dim,
            "x_shape": self.x_shape,
            "dim": self.dim,
            "prior_bound": self.prior_bound,
            "prior_low": self.prior_low.tolist(),
            "prior_high": self.prior_high.tolist(),
            "simulator_scale": self.simulator_scale,
        }
