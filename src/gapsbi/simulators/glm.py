# reference code: https://github.com/sbi-benchmark/sbibm/blob/main/sbibm/tasks/bernoulli_glm/task.py
from typing import Any

import numpy as np

from gapsbi.simulators.base import Simulator


class GLMSimulator(Simulator):
    """Bernoulli GLM simulator with fixed stimulus features."""

    def __init__(
        self,
        dim: int = 10,
        prior_bound: float = 2.0,
        duration: int = 100,
        stimulus_seed: int = 42,
        summary: str = "sufficient",
    ) -> None:
        self.dim = int(dim)
        self.prior_bound = float(prior_bound)
        self.duration = int(duration)
        self.stimulus_seed = int(stimulus_seed)
        self.summary = summary

        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.prior_bound <= 0:
            raise ValueError("prior_bound must be positive")
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        if self.summary not in {"sufficient", "raw"}:
            raise ValueError("summary must be 'sufficient' or 'raw'")

        self.prior_low = np.full(self.dim, -self.prior_bound, dtype=float)
        self.prior_high = np.full(self.dim, self.prior_bound, dtype=float)
        self.stimulus_I = np.random.default_rng(self.stimulus_seed).normal(size=self.duration)
        self.design_matrix = self._build_design_matrix()

    @property
    def name(self) -> str:
        return "glm" if self.summary == "sufficient" else "glm_raw"

    @property
    def theta_dim(self) -> int:
        return self.dim

    @property
    def x_shape(self) -> tuple[int, ...]:
        if self.summary == "sufficient":
            return (self.dim,)
        return (self.duration,)

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

        psi = theta_batch @ self.design_matrix.T
        p = self._sigmoid(psi)
        y = (rng.uniform(size=p.shape) < p).astype(np.int8)

        if self.summary == "raw":
            x = y
        else:
            x = self._summarize(y)

        if not np.all(np.isfinite(x)):
            raise FloatingPointError("GLM simulation produced nonfinite values")

        return x[0] if single else x

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "theta_dim": self.theta_dim,
            "x_shape": self.x_shape,
            "dim": self.dim,
            "duration": self.duration,
            "prior_bound": self.prior_bound,
            "prior_low": self.prior_low.tolist(),
            "prior_high": self.prior_high.tolist(),
            "stimulus_seed": self.stimulus_seed,
            "summary": self.summary,
        }

    def _build_design_matrix(self) -> np.ndarray:
        design_matrix = np.zeros((self.duration, self.dim), dtype=float)
        design_matrix[:, 0] = 1.0

        for j in range(self.dim - 1):
            if j >= self.duration:
                break
            design_matrix[j:, j + 1] = self.stimulus_I[: self.duration - j]

        return design_matrix

    def _summarize(self, y: np.ndarray) -> np.ndarray:
        num_spikes = np.sum(y, axis=1, dtype=float)
        sta = np.empty((y.shape[0], self.dim - 1), dtype=float)

        for lag in range(self.dim - 1):
            if lag >= self.duration:
                sta[:, lag] = 0.0
            else:
                sta[:, lag] = np.sum(y[:, lag:] * self.stimulus_I[: self.duration - lag], axis=1)

        return np.concatenate([num_spikes[:, None], sta], axis=1)

    @staticmethod
    def _sigmoid(psi: np.ndarray) -> np.ndarray:
        psi = np.clip(psi, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-psi))
