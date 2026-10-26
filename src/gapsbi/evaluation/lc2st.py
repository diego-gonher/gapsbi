"""Small, artifact-oriented helpers for standard L-C2ST pilots.

These functions intentionally know nothing about estimator checkpoints.  They
validate and select from already-saved posterior draws, which keeps diagnostic
experiments separate from GapSBI's reference-evaluation pipeline.
"""
from __future__ import annotations

import numpy as np


def mask_augmented_context(x_obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return ``[x_obs, mask]`` after checking the persisted representation."""
    x_obs = np.asarray(x_obs)
    mask = np.asarray(mask)
    if x_obs.shape != mask.shape:
        raise ValueError(f"x_obs shape {x_obs.shape} does not match mask {mask.shape}.")
    if x_obs.ndim != 2:
        raise ValueError("Expected a batch of two-dimensional observations.")
    if not np.all(np.isfinite(x_obs)):
        raise ValueError("Observed data contain non-finite values.")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("Mask must be binary.")
    return np.concatenate((x_obs, mask.astype(x_obs.dtype, copy=False)), axis=1)


def select_one_finite_draw_per_observation(
    posterior_samples: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Select one finite draw per observation without changing saved samples."""
    samples = np.asarray(posterior_samples)
    if samples.ndim != 3:
        raise ValueError("posterior_samples must have shape (observations, draws, theta_dim).")
    valid = np.isfinite(samples).all(axis=2)
    if not np.all(valid.any(axis=1)):
        bad = np.flatnonzero(~valid.any(axis=1)).tolist()
        raise ValueError(f"No finite posterior draw for observations {bad}.")
    rng = np.random.default_rng(seed)
    selected = np.empty((samples.shape[0], samples.shape[2]), dtype=samples.dtype)
    draw_indices = np.empty(samples.shape[0], dtype=np.int64)
    for index, row_valid in enumerate(valid):
        choices = np.flatnonzero(row_valid)
        draw = int(rng.choice(choices))
        selected[index] = samples[index, draw]
        draw_indices[index] = draw
    return selected, draw_indices


def derangement(size: int, seed: int) -> np.ndarray:
    """Return a reproducible permutation with no fixed points for a control."""
    if size < 2:
        raise ValueError("A derangement requires at least two observations.")
    rng = np.random.default_rng(seed)
    identity = np.arange(size)
    # The expected number of attempts is e; retaining this simple rejection
    # sampler makes the returned permutation unambiguously a derangement of
    # the original observation indices.
    while True:
        order = rng.permutation(size)
        if np.all(order != identity):
            return order


def farthest_reference_indices(samples: np.ndarray) -> np.ndarray:
    """For each reference posterior, choose the farthest other posterior mean.

    This is solely for a deliberately strong, clearly labeled L-C2ST positive
    control; it is not used in benchmark inference.
    """
    samples = np.asarray(samples)
    if samples.ndim != 3 or samples.shape[0] < 2:
        raise ValueError("Expected at least two reference posterior sample blocks.")
    means = samples.mean(axis=1)
    scale = np.maximum(means.std(axis=0), np.finfo(means.dtype).eps)
    distances = np.square((means[:, None, :] - means[None, :, :]) / scale).sum(axis=-1)
    np.fill_diagonal(distances, -np.inf)
    return distances.argmax(axis=1).astype(np.int64)
