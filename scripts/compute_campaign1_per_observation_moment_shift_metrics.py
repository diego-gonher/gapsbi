from __future__ import annotations

import argparse
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from compute_campaign1_per_observation_shift_metrics import (
    load_posterior_array,
    subsample_indices,
    subsample_rows,
    validate_matched_posterior_shapes,
)
from compute_campaign1_shift_metrics import (
    add_reference_input_argument,
    add_reference_metadata,
    get_full_reference_row,
    load_master_csv,
    output_columns,
    prepare_reference_rows,
    write_csv,
)


OBSERVATION_COLUMNS = [
    "problem",
    "method",
    "missingness_type",
    "missing_fraction",
    "seed",
    "observation_index",
    "euclidean_mean_shift",
    "mahalanobis_mean_shift",
    "cov_trace_ratio",
    "logdet_cov_ratio",
]

SUMMARY_COLUMNS = [
    "problem",
    "method",
    "missingness_type",
    "missing_fraction",
    "count",
    "euclidean_mean_shift_mean",
    "euclidean_mean_shift_median",
    "euclidean_mean_shift_std",
    "euclidean_mean_shift_min",
    "euclidean_mean_shift_max",
    "mahalanobis_mean_shift_mean",
    "mahalanobis_mean_shift_median",
    "mahalanobis_mean_shift_std",
    "mahalanobis_mean_shift_min",
    "mahalanobis_mean_shift_max",
    "cov_trace_ratio_mean",
    "cov_trace_ratio_median",
    "cov_trace_ratio_std",
    "cov_trace_ratio_min",
    "cov_trace_ratio_max",
    "logdet_cov_ratio_mean",
    "logdet_cov_ratio_median",
    "logdet_cov_ratio_std",
    "logdet_cov_ratio_min",
    "logdet_cov_ratio_max",
]

METRIC_COLUMNS = [
    "euclidean_mean_shift",
    "mahalanobis_mean_shift",
    "cov_trace_ratio",
    "logdet_cov_ratio",
]


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    rows = load_master_csv(args.input)
    reference_rows = load_master_csv(args.reference_input) if args.reference_input else None
    missing_rows, full_rows = prepare_reference_rows(rows, reference_rows=reference_rows)
    cross_input_reference = args.reference_input is not None

    print("Campaign 1 per-observation posterior moment shift metrics")
    print(f"Input CSV: {args.input}")
    if cross_input_reference:
        print("Reference mode: cross-input")
        print(f"Reference input CSV: {args.reference_input}")
        print(f"Reference CSV rows: {len(reference_rows or [])}")
    else:
        print("Reference mode: same-input")
    print(f"Master CSV rows: {len(rows)}")
    print(f"Missing-data runs to process: {len(missing_rows)}")
    print(f"Full-data reference runs available: {len(full_rows)}")
    print(f"Max observations: {args.max_observations}")
    print(f"Max samples per observation: {args.max_samples_per_observation}")
    print(f"Jitter: {args.jitter}")
    print(f"Seed: {args.seed}")
    print(f"Observation-level output: {args.out}")
    print(f"Summary output: {args.summary_out}")

    observation_rows = build_observation_metric_rows(
        rows,
        reference_rows=reference_rows,
        reference_input=args.reference_input,
        max_observations=args.max_observations,
        max_samples_per_observation=args.max_samples_per_observation,
        jitter=args.jitter,
        seed=args.seed,
    )
    summary_rows = build_summary_rows(observation_rows)

    observation_columns = output_columns(OBSERVATION_COLUMNS, include_reference_metadata=cross_input_reference)
    write_csv(args.out, observation_rows, observation_columns)
    write_csv(args.summary_out, summary_rows, SUMMARY_COLUMNS)

    total_runtime_sec = time.perf_counter() - start_time
    print("Campaign 1 per-observation posterior moment shift metrics complete")
    print(f"Observation rows: {len(observation_rows)}")
    print(f"Summary rows: {len(summary_rows)}")
    print(f"Total runtime: {total_runtime_sec:.2f} sec")
    print(f"Observation-level output: {args.out}")
    print(f"Summary output: {args.summary_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-observation posterior moment shift metrics for Campaign 1.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/campaign1_master_results.csv"),
        help="Path to campaign1_master_results.csv.",
    )
    add_reference_input_argument(parser)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/analysis/campaign1_per_observation_moment_shift_metrics.csv"),
        help="Path for observation-level moment shift metrics.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("outputs/analysis/campaign1_per_observation_moment_shift_metrics_summary.csv"),
        help="Path for grouped moment shift metric summaries.",
    )
    parser.add_argument(
        "--max-observations",
        type=int,
        default=100,
        help="Maximum test observations to compare per run.",
    )
    parser.add_argument(
        "--max-samples-per-observation",
        type=int,
        default=500,
        help="Maximum posterior samples to use per observation and distribution.",
    )
    parser.add_argument("--jitter", type=float, default=1e-6, help="Diagonal covariance jitter.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for subsampling.")
    return parser.parse_args()


def build_observation_metric_rows(
    rows: list[dict[str, Any]],
    max_observations: int,
    max_samples_per_observation: int,
    jitter: float,
    seed: int,
    reference_rows: list[dict[str, Any]] | None = None,
    reference_input: Path | None = None,
) -> list[dict[str, Any]]:
    if max_observations < 1:
        raise ValueError("--max-observations must be at least 1.")
    if max_samples_per_observation < 2:
        raise ValueError("--max-samples-per-observation must be at least 2.")
    if jitter <= 0:
        raise ValueError("--jitter must be positive.")

    missing_rows, full_rows = prepare_reference_rows(rows, reference_rows=reference_rows)

    observation_rows: list[dict[str, Any]] = []
    progress = tqdm(missing_rows, desc="Computing moment shifts", total=len(missing_rows))
    for index, missing_row in enumerate(progress):
        progress.set_postfix(
            problem=missing_row.get("problem", ""),
            method=missing_row.get("method", ""),
            missingness=missing_row.get("missingness", ""),
            epsilon=missing_row.get("epsilon", ""),
        )
        rng = np.random.default_rng(seed + index)
        try:
            observation_rows.extend(
                build_run_observation_metric_rows(
                    missing_row,
                    full_rows,
                    max_observations=max_observations,
                    max_samples_per_observation=max_samples_per_observation,
                    jitter=jitter,
                    rng=rng,
                    reference_input=reference_input,
                )
            )
        except Exception as exc:
            tqdm.write(
                "Error computing moment shifts for "
                f"{missing_row.get('method', '')} {missing_row.get('problem', '')} "
                f"{missing_row.get('missingness', '')} eps={missing_row.get('epsilon', '')} "
                f"seed={missing_row.get('seed', '')}: {exc}"
            )
    return observation_rows


def build_run_observation_metric_rows(
    missing_row: dict[str, Any],
    full_rows: dict[tuple[str, str], dict[str, Any]],
    max_observations: int,
    max_samples_per_observation: int,
    jitter: float,
    rng: np.random.Generator,
    reference_input: Path | None = None,
) -> list[dict[str, Any]]:
    full_row = get_full_reference_row(missing_row, full_rows)
    missing_path = Path(str(missing_row["run_dir"])) / "posterior_samples.h5"
    full_path = Path(str(full_row["run_dir"])) / "posterior_samples.h5"
    missing_post, missing_key = load_posterior_array(missing_path)
    full_post, full_key = load_posterior_array(full_path)

    if missing_key != full_key:
        raise ValueError(f"Posterior keys differ: missing={missing_key}, full={full_key}.")
    validate_matched_posterior_shapes(missing_post, full_post)

    num_eval, num_posterior_samples, _theta_dim = missing_post.shape
    observation_indices = subsample_indices(num_eval, max_observations, rng)
    sample_count = min(max_samples_per_observation, num_posterior_samples, full_post.shape[1])

    rows = []
    for observation_index in observation_indices:
        missing_samples = subsample_rows(missing_post[observation_index], sample_count, rng)
        full_samples = subsample_rows(full_post[observation_index], sample_count, rng)
        try:
            metrics = compute_moment_shift_metrics(
                full_samples=full_samples,
                missing_samples=missing_samples,
                jitter=jitter,
            )
        except Exception as exc:
            tqdm.write(
                "Skipping invalid moment metrics for "
                f"{missing_row.get('method', '')} {missing_row.get('problem', '')} "
                f"{missing_row.get('missingness', '')} eps={missing_row.get('epsilon', '')} "
                f"seed={missing_row.get('seed', '')} obs={int(observation_index)}: {exc}"
            )
            continue

        row = {
            "problem": missing_row["problem"],
            "method": missing_row["method"],
            "missingness_type": missing_row["missingness"],
            "missing_fraction": missing_row["epsilon"],
            "seed": missing_row["seed"],
            "observation_index": int(observation_index),
            **metrics,
        }
        add_reference_metadata(row, reference_input=reference_input, reference_row=full_row)
        rows.append(row)
    return rows


def compute_moment_shift_metrics(
    full_samples: np.ndarray,
    missing_samples: np.ndarray,
    jitter: float = 1e-6,
) -> dict[str, float]:
    full_samples = np.asarray(full_samples, dtype=np.float64)
    missing_samples = np.asarray(missing_samples, dtype=np.float64)
    if full_samples.ndim != 2 or missing_samples.ndim != 2 or full_samples.shape[1] != missing_samples.shape[1]:
        raise ValueError("Posterior sample arrays must be 2D with matching theta dimensions.")
    if full_samples.shape[0] < 2 or missing_samples.shape[0] < 2:
        raise ValueError("Need at least two posterior samples per distribution.")

    mu_full = full_samples.mean(axis=0)
    mu_missing = missing_samples.mean(axis=0)
    delta_mu = mu_missing - mu_full
    euclidean_mean_shift = float(np.linalg.norm(delta_mu))

    cov_full = covariance_matrix(full_samples)
    cov_missing = covariance_matrix(missing_samples)
    jittered_cov_full = jitter_covariance(cov_full, jitter)
    jittered_cov_missing = jitter_covariance(cov_missing, jitter)

    mahalanobis_sq = solve_quadratic_form(jittered_cov_full, delta_mu)
    if not math.isfinite(mahalanobis_sq) or mahalanobis_sq < 0.0:
        raise ValueError(f"Invalid Mahalanobis quadratic form: {mahalanobis_sq}.")
    mahalanobis_mean_shift = float(math.sqrt(max(0.0, mahalanobis_sq)))

    sign_full, logdet_full = np.linalg.slogdet(jittered_cov_full)
    sign_missing, logdet_missing = np.linalg.slogdet(jittered_cov_missing)
    if sign_full <= 0 or sign_missing <= 0 or not math.isfinite(logdet_full) or not math.isfinite(logdet_missing):
        raise ValueError(
            f"Invalid covariance log determinant signs/values: "
            f"full=({sign_full}, {logdet_full}), missing=({sign_missing}, {logdet_missing})."
        )
    logdet_cov_ratio = float(logdet_missing - logdet_full)

    trace_full = float(np.trace(cov_full))
    trace_missing = float(np.trace(cov_missing))
    if not math.isfinite(trace_full) or trace_full <= 0.0 or not math.isfinite(trace_missing):
        raise ValueError(f"Invalid covariance traces: full={trace_full}, missing={trace_missing}.")
    cov_trace_ratio = float(trace_missing / trace_full)

    return {
        "euclidean_mean_shift": euclidean_mean_shift,
        "mahalanobis_mean_shift": mahalanobis_mean_shift,
        "cov_trace_ratio": cov_trace_ratio,
        "logdet_cov_ratio": logdet_cov_ratio,
    }


def covariance_matrix(samples: np.ndarray) -> np.ndarray:
    cov = np.cov(samples, rowvar=False)
    if cov.ndim == 0:
        cov = np.asarray([[float(cov)]], dtype=np.float64)
    return np.asarray(cov, dtype=np.float64)


def jitter_covariance(cov: np.ndarray, jitter: float) -> np.ndarray:
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"Covariance must be square, got shape {cov.shape}.")
    return cov + jitter * np.eye(cov.shape[0], dtype=np.float64)


def solve_quadratic_form(cov: np.ndarray, delta: np.ndarray) -> float:
    try:
        solved = np.linalg.solve(cov, delta)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(cov) @ delta
    return float(delta.T @ solved)


def build_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["problem"], row["method"], row["missingness_type"], row["missing_fraction"])].append(row)

    summary_rows = []
    for key, group in sorted(groups.items()):
        summary: dict[str, Any] = {
            "problem": key[0],
            "method": key[1],
            "missingness_type": key[2],
            "missing_fraction": key[3],
            "count": len(group),
        }
        for metric in METRIC_COLUMNS:
            stats = summarize_metric(group, metric)
            for stat_name, value in stats.items():
                summary[f"{metric}_{stat_name}"] = value
        summary_rows.append(summary)
    return summary_rows


def summarize_metric(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    values = []
    for row in rows:
        try:
            value = float(row[metric])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)

    if not values:
        nan = float("nan")
        return {"mean": nan, "median": nan, "std": nan, "min": nan, "max": nan}

    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


if __name__ == "__main__":
    main()
