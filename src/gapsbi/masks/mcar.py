from typing import Any

import numpy as np

from gapsbi.masks.base import MaskGenerator


def _validate_missing_fraction(missing_fraction: float) -> None:
    if not 0.0 <= missing_fraction <= 1.0:
        raise ValueError("missing_fraction must be in [0, 1].")


class PointMCARMask(MaskGenerator):
    """Independent point-wise MCAR masking."""

    def __init__(self, missing_fraction: float) -> None:
        _validate_missing_fraction(missing_fraction)
        self.missing_fraction = float(missing_fraction)

    @property
    def name(self) -> str:
        return "point_mcar"

    def generate(
        self,
        x_full: np.ndarray,
        theta: np.ndarray | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        x = np.asarray(x_full)
        observed_probability = 1.0 - self.missing_fraction
        return (rng.uniform(size=x.shape) < observed_probability).astype(np.int8)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "missing_fraction": self.missing_fraction,
        }


class BlockMCARMask(MaskGenerator):
    """MCAR masking using contiguous missing blocks for 1D time series."""

    def __init__(self, missing_fraction: float, block_size: int) -> None:
        _validate_missing_fraction(missing_fraction)
        if block_size < 1:
            raise ValueError("block_size must be >= 1.")

        self.missing_fraction = float(missing_fraction)
        self.block_size = int(block_size)

    @property
    def name(self) -> str:
        return "block_mcar"

    def generate(
        self,
        x_full: np.ndarray,
        theta: np.ndarray | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        x = np.asarray(x_full)
        if x.ndim == 1:
            return self._generate_1d(x.shape[0], rng)
        if x.ndim == 2:
            return np.stack([self._generate_1d(x.shape[1], rng) for _ in range(x.shape[0])])
        raise ValueError("BlockMCARMask expects x_full with shape (T,) or (N, T).")

    def _generate_1d(self, length: int, rng: np.random.Generator) -> np.ndarray:
        mask = np.ones(length, dtype=np.int8)
        target_missing = int(round(self.missing_fraction * length))

        if target_missing <= 0:
            return mask
        if target_missing >= length:
            return np.zeros(length, dtype=np.int8)

        max_start = max(length - self.block_size, 0)
        stall_count = 0
        missing_count = 0

        while missing_count < target_missing:
            previous_missing_count = missing_count
            start = int(rng.integers(0, max_start + 1))
            end = min(start + self.block_size, length)
            mask[start:end] = 0
            missing_count = int(np.sum(mask == 0))

            if missing_count == previous_missing_count:
                stall_count += 1
            else:
                stall_count = 0

            if stall_count > length * 10:
                observed = np.flatnonzero(mask == 1)
                if observed.size == 0:
                    break
                start = min(max(int(observed[0]), 0), max_start)
                end = min(start + self.block_size, length)
                mask[start:end] = 0
                missing_count = int(np.sum(mask == 0))
                stall_count = 0

        return mask

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "missing_fraction": self.missing_fraction,
            "block_size": self.block_size,
        }
