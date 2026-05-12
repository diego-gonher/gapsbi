from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class MaskGenerator(ABC):
    """Abstract interface for missing-data mask generators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable mask generator name."""

    @abstractmethod
    def generate(
        self,
        x_full: np.ndarray,
        theta: np.ndarray | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate a binary mask with 1 for observed and 0 for missing."""

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return serializable mask generator metadata."""
