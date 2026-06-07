from __future__ import annotations

import numpy as np

from gapsbi.datasets import apply_train_val_sample_limits


def _make_dataset() -> dict[str, dict[str, np.ndarray]]:
    return {
        split: {
            "theta": np.arange(n * 2).reshape(n, 2) + offset,
            "x_full": np.arange(n * 3).reshape(n, 3) + offset,
            "x_obs": np.arange(n * 3).reshape(n, 3) + offset,
            "mask": np.ones((n, 3), dtype=np.int8),
        }
        for split, n, offset in [
            ("train", 5, 0),
            ("val", 4, 100),
            ("test", 3, 200),
        ]
    }


def test_apply_train_val_sample_limits_default_is_noop() -> None:
    dataset = _make_dataset()

    limited = apply_train_val_sample_limits(dataset)

    assert limited is dataset


def test_apply_train_val_sample_limits_truncates_train_and_val() -> None:
    dataset = _make_dataset()

    limited = apply_train_val_sample_limits(
        dataset,
        max_train_samples=2,
        max_val_samples=3,
    )

    assert limited["train"]["theta"].shape == (2, 2)
    assert limited["val"]["theta"].shape == (3, 2)
    np.testing.assert_array_equal(limited["train"]["theta"], dataset["train"]["theta"][:2])
    np.testing.assert_array_equal(limited["val"]["x_full"], dataset["val"]["x_full"][:3])


def test_apply_train_val_sample_limits_does_not_truncate_test() -> None:
    dataset = _make_dataset()

    limited = apply_train_val_sample_limits(
        dataset,
        max_train_samples=2,
        max_val_samples=2,
    )

    assert limited["test"]["theta"].shape == dataset["test"]["theta"].shape
    np.testing.assert_array_equal(limited["test"]["theta"], dataset["test"]["theta"])
    np.testing.assert_array_equal(limited["test"]["x_full"], dataset["test"]["x_full"])
    np.testing.assert_array_equal(limited["test"]["x_obs"], dataset["test"]["x_obs"])
    np.testing.assert_array_equal(limited["test"]["mask"], dataset["test"]["mask"])
