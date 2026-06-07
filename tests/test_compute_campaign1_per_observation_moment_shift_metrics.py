from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_moment_shift_module():
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    script_path = scripts_dir / "compute_campaign1_per_observation_moment_shift_metrics.py"
    spec = importlib.util.spec_from_file_location("compute_campaign1_per_observation_moment_shift_metrics", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _base_samples() -> np.ndarray:
    return np.asarray(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-2.0, 0.0],
            [2.0, 0.0],
            [0.0, -2.0],
            [0.0, 2.0],
        ],
        dtype=float,
    )


def test_moment_shift_identical_posteriors() -> None:
    metrics = _load_moment_shift_module()
    samples = _base_samples()

    result = metrics.compute_moment_shift_metrics(samples, samples.copy(), jitter=1e-8)

    assert abs(result["euclidean_mean_shift"]) < 1e-12
    assert abs(result["mahalanobis_mean_shift"]) < 1e-12
    assert abs(result["cov_trace_ratio"] - 1.0) < 1e-12
    assert abs(result["logdet_cov_ratio"]) < 1e-12


def test_moment_shift_shifted_gaussian_means() -> None:
    metrics = _load_moment_shift_module()
    full_samples = _base_samples()
    missing_samples = full_samples + np.asarray([1.0, 0.5])

    result = metrics.compute_moment_shift_metrics(full_samples, missing_samples, jitter=1e-8)

    assert result["euclidean_mean_shift"] > 0.0
    assert result["mahalanobis_mean_shift"] > 0.0
    assert abs(result["cov_trace_ratio"] - 1.0) < 1e-12
    assert abs(result["logdet_cov_ratio"]) < 1e-12


def test_moment_shift_larger_covariance_same_mean() -> None:
    metrics = _load_moment_shift_module()
    full_samples = _base_samples()
    missing_samples = 2.0 * full_samples

    result = metrics.compute_moment_shift_metrics(full_samples, missing_samples, jitter=1e-8)

    assert abs(result["euclidean_mean_shift"]) < 1e-12
    assert abs(result["mahalanobis_mean_shift"]) < 1e-12
    assert result["cov_trace_ratio"] > 1.0
    assert result["logdet_cov_ratio"] > 0.0
