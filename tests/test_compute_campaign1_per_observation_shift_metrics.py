from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_per_observation_shift_module():
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    script_path = scripts_dir / "compute_campaign1_per_observation_shift_metrics.py"
    spec = importlib.util.spec_from_file_location("compute_campaign1_per_observation_shift_metrics", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_subsample_indices_is_deterministic_and_sorted() -> None:
    metrics = _load_per_observation_shift_module()
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    indices_a = metrics.subsample_indices(total=20, max_count=5, rng=rng_a)
    indices_b = metrics.subsample_indices(total=20, max_count=5, rng=rng_b)

    np.testing.assert_array_equal(indices_a, indices_b)
    assert indices_a.tolist() == sorted(indices_a.tolist())
    assert len(indices_a) == 5


def test_subsample_indices_returns_all_when_under_limit() -> None:
    metrics = _load_per_observation_shift_module()
    rng = np.random.default_rng(0)

    indices = metrics.subsample_indices(total=4, max_count=10, rng=rng)

    np.testing.assert_array_equal(indices, np.arange(4))


def test_validate_matched_posterior_shapes_accepts_matching_eval_and_theta_dim() -> None:
    metrics = _load_per_observation_shift_module()
    missing = np.zeros((3, 5, 2))
    full = np.zeros((3, 7, 2))

    metrics.validate_matched_posterior_shapes(missing, full)


def test_validate_matched_posterior_shapes_rejects_num_eval_mismatch() -> None:
    metrics = _load_per_observation_shift_module()
    missing = np.zeros((3, 5, 2))
    full = np.zeros((4, 5, 2))

    with pytest.raises(ValueError, match="num_eval differs"):
        metrics.validate_matched_posterior_shapes(missing, full)


def test_validate_matched_posterior_shapes_rejects_theta_dim_mismatch() -> None:
    metrics = _load_per_observation_shift_module()
    missing = np.zeros((3, 5, 2))
    full = np.zeros((3, 5, 3))

    with pytest.raises(ValueError, match="theta_dim differs"):
        metrics.validate_matched_posterior_shapes(missing, full)
