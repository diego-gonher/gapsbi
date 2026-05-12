"""Simple prior distributions for GAPSBI simulators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class UniformPrior:
    """Independent bounded uniform prior."""

    low: np.ndarray
    high: np.ndarray

    def __post_init__(self) -> None:
        low = np.asarray(self.low, dtype=float)
        high = np.asarray(self.high, dtype=float)

        if low.shape != high.shape:
            raise ValueError("low and high must have the same shape.")
        if low.ndim != 1:
            raise ValueError("low and high must be 1D arrays.")
        if np.any(high <= low):
            raise ValueError("All high values must be greater than low values.")

        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    @property
    def dim(self) -> int:
        return int(self.low.shape[0])

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if n < 0:
            raise ValueError("n must be nonnegative.")
        return rng.uniform(self.low, self.high, size=(n, self.dim))

    def log_prob(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)

        if theta.shape[-1] != self.dim:
            raise ValueError(f"Expected last dimension {self.dim}, got {theta.shape[-1]}.")

        inside = np.all((theta >= self.low) & (theta <= self.high), axis=-1)
        volume_log = np.sum(np.log(self.high - self.low))

        return np.where(inside, -volume_log, -np.inf)

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "UniformPrior",
            "low": self.low.tolist(),
            "high": self.high.tolist(),
            "dim": self.dim,
        }


@dataclass(frozen=True)
class LogUniformPrior:
    """Independent log-uniform prior."""

    log_low: np.ndarray
    log_high: np.ndarray

    def __post_init__(self) -> None:
        log_low = np.asarray(self.log_low, dtype=float)
        log_high = np.asarray(self.log_high, dtype=float)

        if log_low.shape != log_high.shape:
            raise ValueError("log_low and log_high must have the same shape.")
        if log_low.ndim != 1:
            raise ValueError("log_low and log_high must be 1D arrays.")
        if np.any(log_high <= log_low):
            raise ValueError("All log_high values must be greater than log_low values.")

        object.__setattr__(self, "log_low", log_low)
        object.__setattr__(self, "log_high", log_high)

    @property
    def dim(self) -> int:
        return int(self.log_low.shape[0])

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if n < 0:
            raise ValueError("n must be nonnegative.")
        log_theta = rng.uniform(self.log_low, self.log_high, size=(n, self.dim))
        return np.exp(log_theta)

    def log_prob(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)

        if theta.shape[-1] != self.dim:
            raise ValueError(f"Expected last dimension {self.dim}, got {theta.shape[-1]}.")

        positive = theta > 0.0
        log_theta = np.where(positive, np.log(theta), np.nan)

        inside = positive & (log_theta >= self.log_low) & (log_theta <= self.log_high)
        inside_all = np.all(inside, axis=-1)

        log_width = np.sum(np.log(self.log_high - self.log_low))
        log_jac = np.sum(log_theta, axis=-1)
        lp = -log_width - log_jac

        return np.where(inside_all, lp, -np.inf)

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "LogUniformPrior",
            "log_low": self.log_low.tolist(),
            "log_high": self.log_high.tolist(),
            "low": np.exp(self.log_low).tolist(),
            "high": np.exp(self.log_high).tolist(),
            "dim": self.dim,
        }
