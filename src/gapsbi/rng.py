"""Random-number utilities for GAPSBI.

All randomness should use explicit numpy.random.Generator objects.
Do not use global np.random state.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


MAX_SEED = 2**63 - 1


def make_rng(seed: int | np.integer | None) -> np.random.Generator:
    """Create a NumPy Generator from a seed."""
    return np.random.default_rng(seed)


def split_rng(rng: np.random.Generator, n: int) -> list[np.random.Generator]:
    """Split one Generator into n independent child Generators."""
    if n < 0:
        raise ValueError("n must be nonnegative.")

    seeds = rng.integers(0, MAX_SEED, size=n, dtype=np.int64)
    return [np.random.default_rng(int(seed)) for seed in seeds]


def make_rngs(
    seed: int | np.integer | None,
    names: Sequence[str],
) -> dict[str, np.random.Generator]:
    """Create named child RNGs from a master seed."""
    master_rng = make_rng(seed)
    child_rngs = split_rng(master_rng, len(names))
    return dict(zip(names, child_rngs, strict=True))
