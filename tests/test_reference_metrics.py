from __future__ import annotations

import math

import numpy as np
import pytest

from gapsbi.evaluation.reference_metrics import (
    aggregate_reference_metrics,
    c2st_accuracy,
    compute_reference_metrics,
    covariance_trace_ratio,
    posterior_mean_shift,
)


def test_posterior_mean_shift_is_euclidean_mean_distance() -> None:
    reference = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    estimator = reference + np.array([3.0, 4.0])

    assert posterior_mean_shift(estimator, reference) == pytest.approx(5.0)


def test_covariance_trace_ratio_uses_estimator_over_reference() -> None:
    reference = np.array([[-1.0], [0.0], [1.0]])
    estimator = 2.0 * reference

    assert covariance_trace_ratio(estimator, reference) == pytest.approx(4.0)


def test_c2st_accuracy_is_near_chance_for_identical_samples() -> None:
    rng = np.random.default_rng(123)
    samples = rng.normal(size=(40, 2))

    score = c2st_accuracy(samples, samples.copy(), seed=7, n_folds=5, max_iter=500)

    assert 0.2 <= score <= 0.8


def test_c2st_accuracy_detects_separated_samples() -> None:
    rng = np.random.default_rng(123)
    reference = rng.normal(loc=0.0, scale=0.2, size=(50, 2))
    estimator = rng.normal(loc=3.0, scale=0.2, size=(50, 2))

    score = c2st_accuracy(reference, estimator, seed=7, n_folds=5, max_iter=500)

    assert score > 0.9


def test_compute_reference_metrics_returns_one_row_per_observation() -> None:
    rng = np.random.default_rng(123)
    reference = rng.normal(size=(2, 30, 2))
    estimator = reference + 0.1

    rows = compute_reference_metrics(
        estimator,
        reference,
        seed=5,
        max_samples_per_observation=20,
        c2st_n_folds=5,
        c2st_max_iter=500,
    )

    assert len(rows) == 2
    assert rows[0]["reference_index"] == 0
    assert rows[0]["num_samples_used"] == 20
    assert rows[0]["posterior_mean_shift"] > 0.0
    assert math.isfinite(rows[0]["covariance_trace_ratio"])


def test_compute_reference_metrics_marks_nonfinite_observations_invalid() -> None:
    rng = np.random.default_rng(123)
    reference = rng.normal(size=(2, 30, 2))
    estimator = reference + 0.1
    estimator[1, :, :] = np.nan

    rows = compute_reference_metrics(
        estimator,
        reference,
        seed=5,
        max_samples_per_observation=20,
        c2st_n_folds=5,
        c2st_max_iter=500,
    )

    assert len(rows) == 2
    assert rows[0]["reference_metrics_valid"] is True
    assert rows[1]["reference_metrics_valid"] is False
    assert "finite" in rows[1]["reference_metrics_error"]
    assert math.isnan(rows[1]["c2st_accuracy"])
    assert math.isnan(rows[1]["posterior_mean_shift"])
    assert math.isnan(rows[1]["covariance_trace_ratio"])


def test_aggregate_reference_metrics_groups_and_reports_se() -> None:
    rows = [
        {"method": "a", "seed": 1, "reference_index": 0, "c2st_accuracy": 0.6, "posterior_mean_shift": 1.0, "covariance_trace_ratio": 2.0},
        {"method": "a", "seed": 1, "reference_index": 1, "c2st_accuracy": 0.8, "posterior_mean_shift": 3.0, "covariance_trace_ratio": 4.0},
        {"method": "a", "seed": 2, "reference_index": 0, "c2st_accuracy": 0.7, "posterior_mean_shift": 2.0, "covariance_trace_ratio": 6.0},
    ]

    summaries = aggregate_reference_metrics(rows, group_keys=("method", "seed"))

    assert len(summaries) == 2
    first = summaries[0]
    assert first["method"] == "a"
    assert first["seed"] == 1
    assert first["num_rows"] == 2
    assert first["c2st_accuracy_mean"] == pytest.approx(0.7)
    assert first["posterior_mean_shift_mean"] == pytest.approx(2.0)
    assert first["covariance_trace_ratio_mean"] == pytest.approx(3.0)
    assert math.isfinite(first["c2st_accuracy_se"])
