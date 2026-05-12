from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Simulator(ABC):
    """Abstract interface for GAPSBI simulators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable simulator name."""

    @property
    @abstractmethod
    def theta_dim(self) -> int:
        """Number of simulator parameters."""

    @property
    @abstractmethod
    def x_shape(self) -> tuple[int, ...]:
        """Shape of one simulated observation."""

    @abstractmethod
    def sample_theta(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample n parameters from the simulator prior."""

    @abstractmethod
    def simulate(self, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Simulate observations for one parameter vector or a batch."""

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return serializable simulator metadata."""
