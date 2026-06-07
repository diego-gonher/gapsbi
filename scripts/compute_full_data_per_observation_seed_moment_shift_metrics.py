from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from compute_campaign1_per_observation_moment_shift_metrics import (
    METRIC_COLUMNS,
    compute_moment_shift_metrics,
    summarize_metric,
)
from compute_campaign1_per_observation_shift_metrics import (
    load_posterior_array,
    subsample_indices,
    subsample_rows,
    validate_matched_posterior_shapes,
)
from compute_campaign1_shift_metrics import load_master_csv, write_csv
from compute_full_data_seed_shift_metrics import build_full_data_seed_pairs


OBSERVATION_COLUMNS = [
    "problem",
    "seed_a",
    "seed_b",
    "run_dir_a",
    "run_dir_b",
    "posterior_key",
    "theta_dim",
    "num_eval_total",
    "num_observations_used",
    "num_posterior_samples_total",
    "num_samples_per_observation_used",
    "observation_index",
    "euclidean_mean_shift",
    "mahalanobis_mean_shift",
    "cov_trace_ratio",
    "logdet_cov_ratio",
    "status",
    "error",
]

SUMMARY_COLUMNS = [
    "problem",
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


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    rows = load_master_csv(args.input)
    full_data_rows = [row for row in rows if row.get("method") == "full_data"]
    seed_pairs = build_full_data_seed_pairs(full_data_rows)

    print("Full-data per-observation seed posterior moment shift metrics")
    print(f"Input CSV: {args.input}")
    print(f"Master CSV rows: {len(rows)}")
    print(f"Full-data runs: {len(full_data_rows)}")
    print(f"Seed pairs to process: {len(seed_pairs)}")
    print(f"Max observations: {args.max_observations}")
    print(f"Max samples per observation: {args.max_samples_per_observation}")
    print(f"Jitter: {args.jitter}")
    print(f"Seed: {args.seed}")
    print(f"Observation-level output: {args.out}")
    print(f"Summary output: {args.summary_out}")

    observation_rows = build_observation_metric_rows(
        seed_pairs,
        max_observations=args.max_observations,
        max_samples_per_observation=args.max_samples_per_observation,
        jitter=args.jitter,
        seed=args.seed,
    )
    summary_rows = build_summary_rows(observation_rows)

    write_csv(args.out, observation_rows, OBSERVATION_COLUMNS)
    write_csv(args.summary_out, summary_rows, SUMMARY_COLUMNS)

    ok_rows = sum(row["status"] == "ok" for row in observation_rows)
    failed_rows = len(observation_rows) - ok_rows
    total_runtime_sec = time.perf_counter() - start_time
    print("Full-data per-observation seed posterior moment shift metrics complete")
    print(f"Observation rows: {len(observation_rows)}")
    print(f"Successful rows: {ok_rows}")
    print(f"Failed rows: {failed_rows}")
    print(f"Summary rows: {len(summary_rows)}")
    print(f"Total runtime: {total_runtime_sec:.2f} sec")
    print(f"Observation-level output: {args.out}")
    print(f"Summary output: {args.summary_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-observation posterior moment shifts between full-data NPE seeds.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/campaign1_master_results.csv"),
        help="Path to campaign1_master_results.csv.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/analysis/full_data_per_observation_seed_moment_shift_metrics.csv"),
        help="Path for observation-level full-data seed moment shift metrics.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("outputs/analysis/full_data_per_observation_seed_moment_shift_metrics_summary.csv"),
        help="Path for grouped full-data seed moment shift summaries.",
    )
    parser.add_argument(
        "--max-observations",
        type=int,
        default=100,
        help="Maximum test observations to compare per seed pair.",
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
    seed_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    max_observations: int,
    max_samples_per_observation: int,
    jitter: float,
    seed: int,
) -> list[dict[str, Any]]:
    if max_observations < 1:
        raise ValueError("--max-observations must be at least 1.")
    if max_samples_per_observation < 2:
        raise ValueError("--max-samples-per-observation must be at least 2.")
    if jitter <= 0:
        raise ValueError("--jitter must be positive.")

    observation_rows: list[dict[str, Any]] = []
    progress = tqdm(seed_pairs, desc="Computing full-data moment shifts", total=len(seed_pairs))
    for index, (row_a, row_b) in enumerate(progress):
        progress.set_postfix(problem=row_a.get("problem", ""), seed_a=row_a.get("seed", ""), seed_b=row_b.get("seed", ""))
        rng = np.random.default_rng(seed + index)
        pair_rows = build_seed_pair_observation_rows(
            row_a,
            row_b,
            max_observations=max_observations,
            max_samples_per_observation=max_samples_per_observation,
            jitter=jitter,
            rng=rng,
        )
        for row in pair_rows:
            if row["status"] == "error":
                tqdm.write(
                    "Error computing full-data seed moment shift for "
                    f"{row['problem']} seed_a={row['seed_a']} seed_b={row['seed_b']} "
                    f"obs={row['observation_index']}: {row['error']}"
                )
        observation_rows.extend(pair_rows)
    return observation_rows


def build_seed_pair_observation_rows(
    row_a: dict[str, Any],
    row_b: dict[str, Any],
    max_observations: int,
    max_samples_per_observation: int,
    jitter: float,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    base_metadata = {
        "problem": row_a.get("problem", ""),
        "seed_a": row_a.get("seed", ""),
        "seed_b": row_b.get("seed", ""),
        "run_dir_a": row_a.get("run_dir", ""),
        "run_dir_b": row_b.get("run_dir", ""),
        "posterior_key": "",
        "theta_dim": "",
        "num_eval_total": "",
        "num_observations_used": "",
        "num_posterior_samples_total": "",
        "num_samples_per_observation_used": "",
    }

    try:
        posterior_path_a = Path(str(row_a["run_dir"])) / "posterior_samples.h5"
        posterior_path_b = Path(str(row_b["run_dir"])) / "posterior_samples.h5"
        posterior_a, key_a = load_posterior_array(posterior_path_a)
        posterior_b, key_b = load_posterior_array(posterior_path_b)

        if key_a != key_b:
            raise ValueError(f"Posterior keys differ: seed_a={key_a}, seed_b={key_b}.")
        validate_matched_posterior_shapes(posterior_a, posterior_b)
    except Exception as exc:
        return [error_row(base_metadata, observation_index="", error=str(exc))]

    num_eval, num_posterior_samples, theta_dim = posterior_a.shape
    observation_indices = subsample_indices(num_eval, max_observations, rng)
    sample_count = min(max_samples_per_observation, num_posterior_samples, posterior_b.shape[1])
    base_metadata.update(
        {
            "posterior_key": key_a,
            "theta_dim": int(theta_dim),
            "num_eval_total": int(num_eval),
            "num_observations_used": int(observation_indices.size),
            "num_posterior_samples_total": int(num_posterior_samples),
            "num_samples_per_observation_used": int(sample_count),
        }
    )

    rows = []
    for observation_index in observation_indices:
        samples_a = subsample_rows(posterior_a[observation_index], sample_count, rng)
        samples_b = subsample_rows(posterior_b[observation_index], sample_count, rng)
        try:
            metrics = compute_moment_shift_metrics(
                full_samples=samples_a,
                missing_samples=samples_b,
                jitter=jitter,
            )
        except Exception as exc:
            rows.append(error_row(base_metadata, observation_index=int(observation_index), error=str(exc)))
            continue

        rows.append(
            {
                **base_metadata,
                "observation_index": int(observation_index),
                **metrics,
                "status": "ok",
                "error": "",
            }
        )
    return rows


def error_row(base_metadata: dict[str, Any], observation_index: int | str, error: str) -> dict[str, Any]:
    return {
        **base_metadata,
        "observation_index": observation_index,
        "euclidean_mean_shift": "",
        "mahalanobis_mean_shift": "",
        "cov_trace_ratio": "",
        "logdet_cov_ratio": "",
        "status": "error",
        "error": error,
    }


def build_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_problem[str(row["problem"])].append(row)

    summary_rows = []
    for problem, group in sorted(rows_by_problem.items()):
        ok_group = [row for row in group if row["status"] == "ok"]
        summary: dict[str, Any] = {
            "problem": problem,
            "count": len(ok_group),
        }
        for metric in METRIC_COLUMNS:
            stats = summarize_metric(ok_group, metric)
            for stat_name, value in stats.items():
                summary[f"{metric}_{stat_name}"] = value
        summary_rows.append(summary)
    return summary_rows


if __name__ == "__main__":
    main()
