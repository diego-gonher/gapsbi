# reference code: https://github.com/Aalto-QuML/RISE/blob/main/simulators/oup.py
from typing import Any

import numpy as np

from gapsbi.simulators.base import Simulator


class OUPSimulator(Simulator):
    """Ornstein-Uhlenbeck process simulator based on the RISE benchmark."""

    def __init__(
        self,
        n: int = 25,
        T: float = 5.0,
        var: float = 0.1,
        y0: float = 10.0,
        prior_low: np.ndarray | None = None,
        prior_high: np.ndarray | None = None,
    ) -> None:
        self.n = int(n)
        self.T = float(T)
        self.var = float(var)
        self.y0 = float(y0)
        self.prior_low = (
            np.array([0.0, -2.0], dtype=float)
            if prior_low is None
            else np.asarray(prior_low, dtype=float)
        )
        self.prior_high = (
            np.array([2.0, 3.0], dtype=float)
            if prior_high is None
            else np.asarray(prior_high, dtype=float)
        )

        if self.prior_low.shape != (2,) or self.prior_high.shape != (2,):
            raise ValueError("prior_low and prior_high must have shape (2,)")
        if np.any(self.prior_high <= self.prior_low):
            raise ValueError("prior_high values must be greater than prior_low values")
        if self.n <= 0:
            raise ValueError("n must be positive")
        if self.T <= 0:
            raise ValueError("T must be positive")
        if self.var < 0:
            raise ValueError("var must be nonnegative")

        self.dt = self.T / (self.n + 1)

    @property
    def name(self) -> str:
        return "oup"

    @property
    def theta_dim(self) -> int:
        return 2

    @property
    def x_shape(self) -> tuple[int, ...]:
        return (self.n,)

    def sample_theta(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if n < 0:
            raise ValueError("n must be nonnegative")
        return rng.uniform(self.prior_low, self.prior_high, size=(n, self.theta_dim))

    def simulate(self, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        theta_array = np.asarray(theta, dtype=float)
        single = theta_array.ndim == 1

        if theta_array.ndim < 1 or theta_array.shape[-1] != self.theta_dim:
            raise ValueError("theta must have shape (2,) or (batch, 2)")
        if single:
            theta_batch = theta_array[None, :]
        elif theta_array.ndim == 2:
            theta_batch = theta_array
        else:
            raise ValueError("theta must have shape (2,) or (batch, 2)")

        theta1 = theta_batch[:, 0]
        if np.any(theta1 < 0):
            raise ValueError("theta1 must be nonnegative")
        theta2 = np.exp(theta_batch[:, 1])

        batch_size = theta_batch.shape[0]
        y = np.empty((batch_size, self.n), dtype=float)
        y[:, 0] = self.y0

        for t in range(self.n - 1):
            w_t = rng.normal(loc=0.0, scale=np.sqrt(self.var), size=batch_size)
            mu = theta1 * (theta2 - y[:, t]) * self.dt
            sigma_noise = 0.5 * np.sqrt(self.dt) * w_t
            y[:, t + 1] = y[:, t] + mu + sigma_noise

        if not np.all(np.isfinite(y)):
            raise FloatingPointError("OUP simulation produced nonfinite values")

        return y[0] if single else y

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "theta_dim": self.theta_dim,
            "x_shape": self.x_shape,
            "n": self.n,
            "T": self.T,
            "dt": self.dt,
            "var": self.var,
            "y0": self.y0,
            "prior_low": self.prior_low.tolist(),
            "prior_high": self.prior_high.tolist(),
        }
