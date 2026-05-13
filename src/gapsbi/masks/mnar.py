from typing import Any

import numpy as np

from gapsbi.masks.base import MaskGenerator


ALLOWED_SCORE_TRANSFORMS = {"identity", "log1p", "abs"}


class SelfCensoringMNARMask(MaskGenerator):
    """Value-dependent MNAR self-censoring mask.

    Scores are computed independently for each sample by min/max shifting values
    into [0, 1]. Constant samples use a score of 0.5 everywhere, avoiding
    division instability while still producing value-independent missingness.
    """

    def __init__(
        self,
        missing_fraction: float,
        eps: float = 1e-12,
        score_transform: str = "identity",
    ) -> None:
        if not 0.0 <= missing_fraction <= 1.0:
            raise ValueError("missing_fraction must be in [0, 1].")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        if score_transform not in ALLOWED_SCORE_TRANSFORMS:
            raise ValueError(
                f"score_transform must be one of {sorted(ALLOWED_SCORE_TRANSFORMS)}."
            )

        self.missing_fraction = float(missing_fraction)
        self.eps = float(eps)
        self.score_transform = score_transform

    @property
    def name(self) -> str:
        return "self_censoring_mnar"

    def generate(
        self,
        x_full: np.ndarray,
        theta: np.ndarray | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        x = np.asarray(x_full, dtype=float)
        scoring_values = self._transform_values(x)
        score = self._score(scoring_values)
        p_missing = np.clip(self.missing_fraction * score, 0.0, 1.0)
        return (rng.uniform(size=x.shape) >= p_missing).astype(np.int8)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "missing_fraction": self.missing_fraction,
            "score": "per_sample_minmax_shift",
            "score_transform": self.score_transform,
            "constant_score": 0.5,
            "eps": self.eps,
        }

    def _transform_values(self, x: np.ndarray) -> np.ndarray:
        if self.score_transform == "identity":
            return x
        if self.score_transform == "abs":
            return np.abs(x)
        if np.any(x < 0):
            raise ValueError("log1p scoring requires nonnegative data.")
        return np.log1p(x)

    def _score(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 0:
            return np.full_like(x, 0.5, dtype=float)
        if x.ndim == 1:
            return self._score_one(x)

        flat = x.reshape((x.shape[0], -1))
        scored = np.empty_like(flat, dtype=float)
        for i, row in enumerate(flat):
            scored[i] = self._score_one(row)
        return scored.reshape(x.shape)

    def _score_one(self, x: np.ndarray) -> np.ndarray:
        x_min = np.min(x)
        x_max = np.max(x)
        value_range = x_max - x_min
        if value_range < self.eps:
            return np.full_like(x, 0.5, dtype=float)
        return (x - x_min) / (value_range + self.eps)


ValueDependentMNARMask = SelfCensoringMNARMask
