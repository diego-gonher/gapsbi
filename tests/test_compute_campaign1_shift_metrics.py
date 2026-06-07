from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np


def _load_shift_metrics_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "compute_campaign1_shift_metrics.py"
    spec = importlib.util.spec_from_file_location("compute_campaign1_shift_metrics", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mmd_is_near_zero_for_identical_samples() -> None:
    metrics = _load_shift_metrics_module()
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(80, 3))

    mmd2 = metrics.rbf_mmd2(samples, samples.copy())

    assert mmd2 < 1e-12


def test_mmd_is_larger_for_shifted_samples() -> None:
    metrics = _load_shift_metrics_module()
    rng = np.random.default_rng(1)
    samples = rng.normal(size=(120, 2))
    shifted = samples + 2.0

    same_mmd2 = metrics.rbf_mmd2(samples, samples.copy())
    shifted_mmd2 = metrics.rbf_mmd2(samples, shifted)

    assert shifted_mmd2 > same_mmd2 + 0.1


def test_c2st_is_near_chance_for_same_distribution() -> None:
    metrics = _load_shift_metrics_module()
    rng = np.random.default_rng(2)
    x = rng.normal(size=(300, 4))
    y = rng.normal(size=(300, 4))

    accuracy = metrics.c2st_accuracy(x, y, seed=2)

    assert 0.35 <= accuracy <= 0.65


def test_c2st_is_high_for_clearly_separated_samples() -> None:
    metrics = _load_shift_metrics_module()
    rng = np.random.default_rng(3)
    x = rng.normal(size=(300, 4))
    y = rng.normal(loc=4.0, size=(300, 4))

    accuracy = metrics.c2st_accuracy(x, y, seed=3)

    assert accuracy > 0.9


def test_select_posterior_key_prefers_scaled(tmp_path: Path) -> None:
    metrics = _load_shift_metrics_module()
    path = tmp_path / "posterior_samples.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("theta_posterior", data=np.zeros((2, 3, 1), dtype=np.float32))
        h5.create_dataset("theta_posterior_scaled", data=np.ones((2, 3, 1), dtype=np.float32))

    with h5py.File(path, "r") as h5:
        key = metrics.select_posterior_key(h5)

    assert key == "theta_posterior_scaled"
