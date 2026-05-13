from typing import Any

import numpy as np

from gapsbi.masks.base import MaskGenerator


ALLOWED_COORDINATE_MAR_MODES = {"increasing", "decreasing", "middle"}


class CoordinateMARMask(MaskGenerator):
    """Coordinate-dependent MAR mask for 1D observations.

    Missingness depends only on coordinate/index metadata along the last axis,
    not on the values of x_full.
    """

    def __init__(
        self,
        missing_fraction: float,
        mode: str = "increasing",
        floor: float = 0.05,
        max_probability: float = 0.95,
        middle_width: float = 0.2,
    ) -> None:
        if not 0.0 <= missing_fraction <= 1.0:
            raise ValueError("missing_fraction must be in [0, 1].")
        if mode not in ALLOWED_COORDINATE_MAR_MODES:
            raise ValueError(f"mode must be one of {sorted(ALLOWED_COORDINATE_MAR_MODES)}.")
        if floor < 0.0:
            raise ValueError("floor must be >= 0.")
        if not 0.0 <= max_probability <= 1.0:
            raise ValueError("max_probability must be in [0, 1].")
        if middle_width <= 0.0:
            raise ValueError("middle_width must be > 0.")

        self.missing_fraction = float(missing_fraction)
        self.mode = mode
        self.floor = float(floor)
        self.max_probability = float(max_probability)
        self.middle_width = float(middle_width)

    @property
    def name(self) -> str:
        return "coordinate_mar"

    def generate(
        self,
        x_full: np.ndarray,
        theta: np.ndarray | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        x = np.asarray(x_full)
        if x.ndim not in {1, 2}:
            raise ValueError("CoordinateMARMask expects x_full with shape (D,) or (N, D).")

        p_missing = self._missing_probability(x.shape[-1])
        return (rng.uniform(size=x.shape) >= p_missing).astype(np.int8)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "missing_fraction": self.missing_fraction,
            "mode": self.mode,
            "floor": self.floor,
            "max_probability": self.max_probability,
            "middle_width": self.middle_width,
            "score": "coordinate_index",
        }

    def _missing_probability(self, length: int) -> np.ndarray:
        coord = np.array([0.5], dtype=float) if length == 1 else np.linspace(0.0, 1.0, length)

        if self.mode == "increasing":
            score = self.floor + coord
        elif self.mode == "decreasing":
            score = self.floor + (1.0 - coord)
        else:
            score = self.floor + np.exp(
                -0.5 * ((coord - 0.5) / self.middle_width) ** 2
            )

        score = score / np.mean(score)
        return np.clip(self.missing_fraction * score, 0.0, self.max_probability)
