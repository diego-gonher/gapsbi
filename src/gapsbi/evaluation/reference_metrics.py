from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier


DEFAULT_REFERENCE_METRICS = (
    "c2st_accuracy",
    "posterior_mean_shift",
    "covariance_trace_ratio",
)


def c2st_accuracy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 1,
    test_size: float | None = None,
    n_folds: int = 5,
    z_score: bool = True,
    noise_scale: float | None = None,
    max_iter: int = 10_000,
) -> float:
    """SBIBM-style classifier two-sample test accuracy.

    By default this follows ``sbibm.metrics.c2st``: z-score using the first
    sample set, train an MLP with two ``10 * dim`` hidden layers, and report
    mean 5-fold cross-validation accuracy. If ``test_size`` is provided, a
    single stratified train/test split is used instead.
    """
    x_arr, y_arr = validate_sample_pair(x, y)
    if x_arr.shape[0] < 2 or y_arr.shape[0] < 2:
        raise ValueError("C2ST requires at least two samples per distribution.")

    if z_score:
        mean = x_arr.mean(axis=0, keepdims=True)
        std = x_arr.std(axis=0, ddof=0, keepdims=True)
        std = np.where(std > 0.0, std, 1.0)
        x_arr = (x_arr - mean) / std
        y_arr = (y_arr - mean) / std

    if noise_scale is not None:
        rng = np.random.default_rng(seed)
        x_arr = x_arr + float(noise_scale) * rng.standard_normal(x_arr.shape)
        y_arr = y_arr + float(noise_scale) * rng.standard_normal(y_arr.shape)

    features = np.concatenate((x_arr, y_arr), axis=0)
    labels = np.concatenate(
        (np.zeros(x_arr.shape[0], dtype=int), np.ones(y_arr.shape[0], dtype=int))
    )
    classifier = make_c2st_classifier(theta_dim=x_arr.shape[1], seed=seed, max_iter=max_iter)

    if test_size is not None:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=float(test_size),
            random_state=seed,
            stratify=labels,
        )
        classifier.fit(x_train, y_train)
        return float(classifier.score(x_test, y_test))

    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}.")
    min_class_count = min(x_arr.shape[0], y_arr.shape[0])
    if n_folds > min_class_count:
        raise ValueError(
            f"n_folds={n_folds} exceeds the smaller class size={min_class_count}."
        )
    cv = KFold(n_splits=int(n_folds), shuffle=True, random_state=seed)
    scores = cross_val_score(classifier, features, labels, cv=cv, scoring="accuracy")
    return float(np.asarray(scores, dtype=np.float64).mean())


def make_c2st_classifier(*, theta_dim: int, seed: int, max_iter: int) -> MLPClassifier:
    if theta_dim <= 0:
        raise ValueError(f"theta_dim must be positive, got {theta_dim}.")
    return MLPClassifier(
        activation="relu",
        hidden_layer_sizes=(10 * theta_dim, 10 * theta_dim),
        max_iter=int(max_iter),
        solver="adam",
        random_state=int(seed),
    )


def posterior_mean_shift(samples: np.ndarray, reference_samples: np.ndarray) -> float:
    """Euclidean distance between estimator and reference posterior means."""
    sample_arr, reference_arr = validate_sample_pair(samples, reference_samples)
    delta = sample_arr.mean(axis=0) - reference_arr.mean(axis=0)
    return float(np.linalg.norm(delta))


def covariance_trace_ratio(samples: np.ndarray, reference_samples: np.ndarray) -> float:
    """Ratio ``trace(cov(estimator)) / trace(cov(reference))``."""
    sample_arr, reference_arr = validate_sample_pair(samples, reference_samples)
    sample_trace = covariance_trace(sample_arr)
    reference_trace = covariance_trace(reference_arr)
    if not math.isfinite(reference_trace) or reference_trace <= 0.0:
        raise ValueError(f"Invalid reference covariance trace: {reference_trace}.")
    if not math.isfinite(sample_trace):
        raise ValueError(f"Invalid estimator covariance trace: {sample_trace}.")
    return float(sample_trace / reference_trace)


def compute_reference_metrics(
    estimator_samples: np.ndarray,
    reference_samples: np.ndarray,
    *,
    seed: int = 1,
    max_samples_per_observation: int | None = 10_000,
    c2st_test_size: float | None = None,
    c2st_n_folds: int = 5,
    c2st_max_iter: int = 10_000,
) -> list[dict[str, Any]]:
    """Compute reference metrics separately for each reference observation."""
    estimator = validate_posterior_array(estimator_samples, "estimator_samples")
    reference = validate_posterior_array(reference_samples, "reference_samples")
    validate_matched_reference_arrays(estimator, reference)

    rows: list[dict[str, Any]] = []
    for obs_idx in range(estimator.shape[0]):
        rng = np.random.default_rng(seed + obs_idx)
        sample_count = min(estimator.shape[1], reference.shape[1])
        if max_samples_per_observation is not None:
            if max_samples_per_observation < 2:
                raise ValueError("max_samples_per_observation must be at least 2.")
            sample_count = min(sample_count, int(max_samples_per_observation))
        estimator_obs = subsample_rows(estimator[obs_idx], sample_count, rng)
        reference_obs = subsample_rows(reference[obs_idx], sample_count, rng)

        row = {
            "reference_index": obs_idx,
            "theta_dim": int(estimator.shape[2]),
            "num_estimator_samples_total": int(estimator.shape[1]),
            "num_reference_samples_total": int(reference.shape[1]),
            "num_samples_used": int(sample_count),
            "c2st_accuracy": c2st_accuracy(
                reference_obs,
                estimator_obs,
                seed=seed + obs_idx,
                test_size=c2st_test_size,
                n_folds=c2st_n_folds,
                max_iter=c2st_max_iter,
            ),
            "posterior_mean_shift": posterior_mean_shift(estimator_obs, reference_obs),
            "covariance_trace_ratio": covariance_trace_ratio(estimator_obs, reference_obs),
        }
        rows.append(row)
    return rows


def aggregate_reference_metrics(
    rows: Iterable[dict[str, Any]],
    *,
    group_keys: Sequence[str] = (),
    metric_names: Sequence[str] = DEFAULT_REFERENCE_METRICS,
) -> list[dict[str, Any]]:
    """Aggregate per-reference rows by group, using mean/std/SE over rows.

    For publication tables, first call this with group keys that include
    ``seed`` to average over reference observations within a seed. Then call it
    again without ``reference_index`` and with keys excluding ``seed`` to
    summarize seed-level means across seeds.
    """
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in group_keys)].append(row)

    summaries: list[dict[str, Any]] = []
    for key_values, group in sorted(grouped.items(), key=lambda item: item[0]):
        summary = {key: value for key, value in zip(group_keys, key_values, strict=True)}
        summary["num_rows"] = len(group)
        if "seed" in group_keys:
            summary["num_seeds"] = 1
        else:
            seeds = {row.get("seed") for row in group if row.get("seed", "") != ""}
            summary["num_seeds"] = len(seeds) if seeds else ""

        for metric_name in metric_names:
            values = finite_metric_values(group, metric_name)
            summary[f"{metric_name}_mean"] = mean_or_nan(values)
            summary[f"{metric_name}_std"] = std_or_nan(values)
            summary[f"{metric_name}_se"] = se_or_nan(values)
            summary[f"{metric_name}_median"] = median_or_nan(values)
        summaries.append(summary)
    return summaries


def validate_sample_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.ndim != 2 or y_arr.ndim != 2:
        raise ValueError("Sample arrays must be 2D with shape (num_samples, theta_dim).")
    if x_arr.shape[1] != y_arr.shape[1]:
        raise ValueError(f"Theta dimensions differ: {x_arr.shape[1]} vs {y_arr.shape[1]}.")
    if x_arr.shape[0] < 2 or y_arr.shape[0] < 2:
        raise ValueError("Need at least two samples per distribution.")
    if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
        raise ValueError("Sample arrays must contain only finite values.")
    return x_arr, y_arr


def validate_posterior_array(samples: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"{name} must have shape (num_observations, num_samples, theta_dim).")
    if arr.shape[0] <= 0 or arr.shape[1] < 2 or arr.shape[2] <= 0:
        raise ValueError(f"{name} has invalid shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def validate_matched_reference_arrays(estimator: np.ndarray, reference: np.ndarray) -> None:
    if estimator.shape[0] != reference.shape[0]:
        raise ValueError(
            f"Number of observations differs: estimator={estimator.shape[0]}, "
            f"reference={reference.shape[0]}."
        )
    if estimator.shape[2] != reference.shape[2]:
        raise ValueError(
            f"Theta dimensions differ: estimator={estimator.shape[2]}, "
            f"reference={reference.shape[2]}."
        )


def covariance_trace(samples: np.ndarray) -> float:
    cov = np.cov(np.asarray(samples, dtype=np.float64), rowvar=False)
    if cov.ndim == 0:
        return float(cov)
    return float(np.trace(cov))


def subsample_rows(samples: np.ndarray, sample_count: int, rng: np.random.Generator) -> np.ndarray:
    if samples.shape[0] <= sample_count:
        return samples
    indices = rng.choice(samples.shape[0], size=sample_count, replace=False)
    return samples[indices]


def finite_metric_values(rows: Iterable[dict[str, Any]], metric_name: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row[metric_name])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def mean_or_nan(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def std_or_nan(values: Sequence[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) >= 2 else float("nan")


def se_or_nan(values: Sequence[float]) -> float:
    return float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) >= 2 else float("nan")


def median_or_nan(values: Sequence[float]) -> float:
    return float(np.median(values)) if values else float("nan")
