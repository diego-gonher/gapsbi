"""Pure helpers for direct full- versus partial-reference posterior analyses."""
from __future__ import annotations

import numpy as np


def cross_sample_duplicate_summary(first: np.ndarray, second: np.ndarray) -> dict[str, int]:
    """Count exact cross-set sample duplicates without conflating within-set repeats.

    Exact shared rows invalidate ordinary row-wise C2ST cross-validation because
    identical feature vectors can receive opposing labels in train and test
    folds.  This is intentionally an exact-byte diagnostic: coincidental
    equality of continuous MCMC states is practically negligible, while copied
    posterior arrays are detected reliably.
    """
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("Both samples must be 2D with a shared parameter dimension.")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("Duplicate checks require finite samples.")
    encode = lambda array: [np.ascontiguousarray(row).view(np.uint8).tobytes() for row in array]
    first_rows, second_rows = encode(first), encode(second)
    shared = set(first_rows).intersection(second_rows)
    return {
        "first_rows": len(first_rows),
        "second_rows": len(second_rows),
        "shared_unique_rows": len(shared),
        "shared_first_row_count": sum(row in shared for row in first_rows),
        "shared_second_row_count": sum(row in shared for row in second_rows),
    }


def matched_reference_indices(full_observations: dict, partial_observations: dict) -> np.ndarray:
    """Validate immutable observation identity and return original reference IDs."""
    ids = np.asarray(partial_observations.get("reference_indices", np.arange(len(partial_observations["theta_true"]))), dtype=np.int64)
    if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(full_observations["theta_true"])):
        raise ValueError("Partial reference_indices are invalid for the full reference artifact.")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Partial reference_indices must be unique.")
    for key in ("theta_true", "x_full"):
        if not np.array_equal(partial_observations[key], full_observations[key][ids]):
            raise ValueError(f"Partial {key} does not exactly match its full reference observation.")
    return ids


def bootstrap_interval(values: np.ndarray, *, statistic: str, seed: int, draws: int = 2_000) -> tuple[float, float, float, float]:
    """Return point estimate, two-sided 95% bootstrap interval, and sample SD."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (float("nan"),) * 4
    function = np.mean if statistic == "mean" else np.median if statistic == "median" else None
    if function is None:
        raise ValueError(f"Unsupported statistic: {statistic}")
    point = float(function(values))
    if len(values) == 1:
        return point, float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    replicates = function(rng.choice(values, size=(draws, len(values)), replace=True), axis=1)
    return point, float(np.quantile(replicates, .025)), float(np.quantile(replicates, .975)), float(np.std(values, ddof=1))
