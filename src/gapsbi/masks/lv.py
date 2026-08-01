from typing import Any

import numpy as np

from gapsbi.masks.base import MaskGenerator


ALLOWED_LV_MAR_MODES = {"increasing", "decreasing", "middle"}


class LotkaVolterraTimeMaskMixin:
    num_populations: int

    def _reshape(self, x_full: np.ndarray) -> tuple[np.ndarray, bool]:
        x = np.asarray(x_full)
        if x.ndim == 1:
            if x.shape[0] % self.num_populations != 0:
                raise ValueError("LV time masks expect length divisible by num_populations.")
            return x.reshape((-1, self.num_populations)), True
        if x.ndim == 2:
            if x.shape[1] % self.num_populations != 0:
                raise ValueError("LV time masks expect feature length divisible by num_populations.")
            return x.reshape((x.shape[0], -1, self.num_populations)), False
        raise ValueError("LV time masks expect x_full with shape (D,) or (N, D).")

    def _expand_time_mask(self, time_mask: np.ndarray, single: bool) -> np.ndarray:
        if single:
            return np.repeat(time_mask[:, None], self.num_populations, axis=1).reshape(-1).astype(np.int8)
        return np.repeat(time_mask[:, :, None], self.num_populations, axis=2).reshape(
            time_mask.shape[0], -1
        ).astype(np.int8)


class LotkaVolterraTimeBlockMCARMask(LotkaVolterraTimeMaskMixin, MaskGenerator):
    """Time-block MCAR mask that hides both LV populations at selected times."""

    def __init__(
        self,
        missing_fraction: float,
        block_size: int,
        num_populations: int = 2,
    ) -> None:
        if not 0.0 <= missing_fraction <= 1.0:
            raise ValueError("missing_fraction must be in [0, 1].")
        if block_size < 1:
            raise ValueError("block_size must be >= 1.")
        if num_populations < 1:
            raise ValueError("num_populations must be positive.")
        self.missing_fraction = float(missing_fraction)
        self.block_size = int(block_size)
        self.num_populations = int(num_populations)

    @property
    def name(self) -> str:
        return "lv_time_block_mcar"

    def generate(
        self,
        x_full: np.ndarray,
        theta: np.ndarray | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        x, single = self._reshape(x_full)
        if single:
            time_mask = self._generate_one(x.shape[0], rng)
        else:
            time_mask = np.stack([self._generate_one(x.shape[1], rng) for _ in range(x.shape[0])])
        return self._expand_time_mask(time_mask, single)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "missing_fraction": self.missing_fraction,
            "block_size": self.block_size,
            "num_populations": self.num_populations,
            "mask_unit": "time_slice",
        }

    def _generate_one(self, num_timepoints: int, rng: np.random.Generator) -> np.ndarray:
        mask = np.ones(num_timepoints, dtype=np.int8)
        target_missing = int(round(self.missing_fraction * num_timepoints))
        if target_missing <= 0:
            return mask
        if target_missing >= num_timepoints:
            return np.zeros(num_timepoints, dtype=np.int8)

        missing_count = 0
        stall_count = 0
        while missing_count < target_missing:
            previous_missing_count = missing_count
            block_length = min(self.block_size, target_missing - missing_count)
            max_start = max(num_timepoints - block_length, 0)
            start = int(rng.integers(0, max_start + 1))
            end = min(start + block_length, num_timepoints)
            mask[start:end] = 0
            missing_count = int(np.sum(mask == 0))
            if missing_count == previous_missing_count:
                stall_count += 1
            else:
                stall_count = 0
            if stall_count > num_timepoints * 10:
                observed = np.flatnonzero(mask == 1)
                if observed.size == 0:
                    break
                block_length = min(self.block_size, target_missing - missing_count)
                max_start = max(num_timepoints - block_length, 0)
                start = min(max(int(observed[0]), 0), max_start)
                end = min(start + block_length, num_timepoints)
                mask[start:end] = 0
                missing_count = int(np.sum(mask == 0))
                stall_count = 0
        return mask


class LotkaVolterraTimeMARMask(LotkaVolterraTimeMaskMixin, MaskGenerator):
    """Time-index MAR mask that hides both LV populations at selected times."""

    def __init__(
        self,
        missing_fraction: float,
        mode: str = "increasing",
        floor: float = 0.05,
        max_probability: float = 0.95,
        middle_width: float = 0.2,
        num_populations: int = 2,
    ) -> None:
        if not 0.0 <= missing_fraction <= 1.0:
            raise ValueError("missing_fraction must be in [0, 1].")
        if mode not in ALLOWED_LV_MAR_MODES:
            raise ValueError(f"mode must be one of {sorted(ALLOWED_LV_MAR_MODES)}.")
        if floor < 0.0:
            raise ValueError("floor must be >= 0.")
        if not 0.0 <= max_probability <= 1.0:
            raise ValueError("max_probability must be in [0, 1].")
        if middle_width <= 0.0:
            raise ValueError("middle_width must be > 0.")
        if num_populations < 1:
            raise ValueError("num_populations must be positive.")
        self.missing_fraction = float(missing_fraction)
        self.mode = mode
        self.floor = float(floor)
        self.max_probability = float(max_probability)
        self.middle_width = float(middle_width)
        self.num_populations = int(num_populations)

    @property
    def name(self) -> str:
        return "lv_time_mar"

    def generate(
        self,
        x_full: np.ndarray,
        theta: np.ndarray | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        x, single = self._reshape(x_full)
        num_timepoints = x.shape[0] if single else x.shape[1]
        p_missing = self._missing_probability(num_timepoints)
        if single:
            time_mask = (rng.uniform(size=num_timepoints) >= p_missing).astype(np.int8)
        else:
            time_mask = (
                rng.uniform(size=(x.shape[0], num_timepoints)) >= p_missing[None, :]
            ).astype(np.int8)
        return self._expand_time_mask(time_mask, single)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "missing_fraction": self.missing_fraction,
            "mode": self.mode,
            "floor": self.floor,
            "max_probability": self.max_probability,
            "middle_width": self.middle_width,
            "num_populations": self.num_populations,
            "score": "time_index",
            "mask_unit": "time_slice",
        }

    def _missing_probability(self, num_timepoints: int) -> np.ndarray:
        time = np.array([0.5], dtype=float) if num_timepoints == 1 else np.linspace(0.0, 1.0, num_timepoints)
        if self.mode == "increasing":
            score = self.floor + time
        elif self.mode == "decreasing":
            score = self.floor + (1.0 - time)
        else:
            score = self.floor + np.exp(
                -0.5 * ((time - 0.5) / self.middle_width) ** 2
            )
        score = score / np.mean(score)
        return np.clip(self.missing_fraction * score, 0.0, self.max_probability)


class LotkaVolterraLogTotalMNARMask(LotkaVolterraTimeMaskMixin, MaskGenerator):
    """Time-slice MNAR mask based on log total population."""

    def __init__(
        self,
        missing_fraction: float,
        eps: float = 1e-12,
        num_populations: int = 2,
    ) -> None:
        if not 0.0 <= missing_fraction <= 1.0:
            raise ValueError("missing_fraction must be in [0, 1].")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        if num_populations < 1:
            raise ValueError("num_populations must be positive.")
        self.missing_fraction = float(missing_fraction)
        self.eps = float(eps)
        self.num_populations = int(num_populations)

    @property
    def name(self) -> str:
        return "lv_log_total_mnar"

    def generate(
        self,
        x_full: np.ndarray,
        theta: np.ndarray | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        x, single = self._reshape(x_full)
        totals = np.log(np.sum(x, axis=-1) + self.eps)
        score = self._score(totals)
        p_missing = np.clip(self.missing_fraction * score, 0.0, 1.0)
        time_mask = (rng.uniform(size=p_missing.shape) >= p_missing).astype(np.int8)
        return self._expand_time_mask(time_mask, single)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "missing_fraction": self.missing_fraction,
            "score": "mean_normalized_per_sample_minmax_log_total_population",
            "constant_score": 1.0,
            "eps": self.eps,
            "num_populations": self.num_populations,
            "mask_unit": "time_slice",
        }

    def _score(self, values: np.ndarray) -> np.ndarray:
        if values.ndim == 1:
            return self._score_one(values)
        scored = np.empty_like(values, dtype=float)
        for i, row in enumerate(values):
            scored[i] = self._score_one(row)
        return scored

    def _score_one(self, values: np.ndarray) -> np.ndarray:
        value_min = np.min(values)
        value_max = np.max(values)
        value_range = value_max - value_min
        if value_range < self.eps:
            return np.ones_like(values, dtype=float)
        score = (values - value_min) / (value_range + self.eps)
        score_mean = np.mean(score)
        if score_mean < self.eps:
            return np.ones_like(values, dtype=float)
        return score / score_mean
