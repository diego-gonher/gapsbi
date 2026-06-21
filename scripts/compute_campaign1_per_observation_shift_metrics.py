from __future__ import annotations

import argparse
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm

from compute_campaign1_shift_metrics import (
    add_reference_input_argument,
    add_reference_metadata,
    c2st_accuracy,
    get_full_reference_row,
    load_master_csv,
    mean_metric,
    output_columns,
    prepare_reference_rows,
    rbf_mmd2,
    select_posterior_key,
    std_metric,
    subsample_rows,
    write_csv,
)


SEED_LEVEL_COLUMNS = [
    "method",
    "problem",
    "missingness",
    "epsilon",
    "experiment",
    "seed",
    "missing_run_dir",
    "full_run_dir",
    "posterior_key",
    "theta_dim",
    "num_eval_total",
    "num_observations_used",
    "num_posterior_samples_total",
    "num_samples_per_observation_used",
    "mmd2_rbf_obs_mean",
    "mmd2_rbf_obs_std",
    "mmd2_rbf_obs_median",
    "c2st_accuracy_obs_mean",
    "c2st_accuracy_obs_std",
    "c2st_accuracy_obs_median",
    "status",
    "error",
]

SUMMARY_COLUMNS = [
    "method",
    "problem",
    "missingness",
    "epsilon",
    "num_rows",
    "num_seeds",
    "mmd2_rbf_obs_mean_mean",
    "mmd2_rbf_obs_mean_std",
    "mmd2_rbf_obs_median_mean",
    "mmd2_rbf_obs_median_std",
    "c2st_accuracy_obs_mean_mean",
    "c2st_accuracy_obs_mean_std",
    "c2st_accuracy_obs_median_mean",
    "c2st_accuracy_obs_median_std",
]


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    rows = load_master_csv(args.input)
    reference_rows = load_master_csv(args.reference_input) if args.reference_input else None
    missing_rows, full_rows = prepare_reference_rows(rows, reference_rows=reference_rows)
    cross_input_reference = args.reference_input is not None

    print("Campaign 1 per-observation posterior shift metrics")
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
    print(f"Seed: {args.seed}")
    print(f"Seed-level output: {args.out}")
    print(f"Summary output: {args.summary_out}")

    shift_rows = build_per_observation_shift_rows(
        rows,
        reference_rows=reference_rows,
        reference_input=args.reference_input,
        max_observations=args.max_observations,
        max_samples_per_observation=args.max_samples_per_observation,
        seed=args.seed,
        c2st_test_size=args.c2st_test_size,
    )
    summary_rows = build_summary_rows(shift_rows)

    seed_level_columns = output_columns(SEED_LEVEL_COLUMNS, include_reference_metadata=cross_input_reference)
    write_csv(args.out, shift_rows, seed_level_columns)
    write_csv(args.summary_out, summary_rows, SUMMARY_COLUMNS)

    ok_rows = sum(row["status"] == "ok" for row in shift_rows)
    failed_rows = len(shift_rows) - ok_rows
    total_runtime_sec = time.perf_counter() - start_time
    print("Campaign 1 per-observation posterior shift metrics complete")
    print(f"Successful rows: {ok_rows}")
    print(f"Failed rows: {failed_rows}")
    print(f"Total runtime: {total_runtime_sec:.2f} sec")
    print(f"Seed-level output: {args.out}")
    print(f"Summary output: {args.summary_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-observation MMD and C2ST posterior shift metrics for Campaign 1.",
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
        default=Path("outputs/analysis/campaign1_per_observation_shift_metrics.csv"),
        help="Path for seed-level per-observation shift metrics.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("outputs/analysis/campaign1_per_observation_shift_metrics_summary.csv"),
        help="Path for grouped per-observation shift metric summaries.",
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
    parser.add_argument("--seed", type=int, default=0, help="Random seed for subsampling and C2ST splits.")
    parser.add_argument("--c2st-test-size", type=float, default=0.5, help="Held-out fraction for C2ST.")
    return parser.parse_args()


def build_per_observation_shift_rows(
    rows: list[dict[str, Any]],
    max_observations: int,
    max_samples_per_observation: int,
    seed: int,
    c2st_test_size: float,
    reference_rows: list[dict[str, Any]] | None = None,
    reference_input: Path | None = None,
) -> list[dict[str, Any]]:
    if max_observations < 1:
        raise ValueError("--max-observations must be at least 1.")
    if max_samples_per_observation < 2:
        raise ValueError("--max-samples-per-observation must be at least 2.")

    missing_rows, full_rows = prepare_reference_rows(rows, reference_rows=reference_rows)

    shift_rows = []
    progress = tqdm(missing_rows, desc="Computing per-observation shifts", total=len(missing_rows))
    for index, missing_row in enumerate(progress):
        progress.set_postfix(
            problem=missing_row.get("problem", ""),
            method=missing_row.get("method", ""),
            missingness=missing_row.get("missingness", ""),
            epsilon=missing_row.get("epsilon", ""),
        )
        rng = np.random.default_rng(seed + index)
        row = build_per_observation_shift_row(
            missing_row,
            full_rows,
            max_observations=max_observations,
            max_samples_per_observation=max_samples_per_observation,
            rng=rng,
            c2st_test_size=c2st_test_size,
            seed=seed,
            reference_input=reference_input,
        )
        if row["status"] == "error":
            tqdm.write(
                "Error computing per-observation shift for "
                f"{row['method']} {row['problem']} {row['missingness']} "
                f"eps={row['epsilon']} seed={row['seed']}: {row['error']}"
            )
        shift_rows.append(row)
    return shift_rows


def build_per_observation_shift_row(
    missing_row: dict[str, Any],
    full_rows: dict[tuple[str, str], dict[str, Any]],
    max_observations: int,
    max_samples_per_observation: int,
    rng: np.random.Generator,
    c2st_test_size: float,
    seed: int,
    reference_input: Path | None = None,
) -> dict[str, Any]:
    base_row = {
        "method": missing_row.get("method", ""),
        "problem": missing_row.get("problem", ""),
        "missingness": missing_row.get("missingness", ""),
        "epsilon": missing_row.get("epsilon", ""),
        "experiment": missing_row.get("experiment", ""),
        "seed": missing_row.get("seed", ""),
        "missing_run_dir": missing_row.get("run_dir", ""),
        "full_run_dir": "",
        "posterior_key": "",
        "theta_dim": "",
        "num_eval_total": "",
        "num_observations_used": "",
        "num_posterior_samples_total": "",
        "num_samples_per_observation_used": "",
        "mmd2_rbf_obs_mean": "",
        "mmd2_rbf_obs_std": "",
        "mmd2_rbf_obs_median": "",
        "c2st_accuracy_obs_mean": "",
        "c2st_accuracy_obs_std": "",
        "c2st_accuracy_obs_median": "",
        "status": "error",
        "error": "",
    }
    add_reference_metadata(base_row, reference_input=reference_input)

    try:
        full_row = get_full_reference_row(missing_row, full_rows)
        base_row["full_run_dir"] = full_row.get("run_dir", "")
        add_reference_metadata(base_row, reference_input=reference_input, reference_row=full_row)

        missing_path = Path(str(missing_row["run_dir"])) / "posterior_samples.h5"
        full_path = Path(str(full_row["run_dir"])) / "posterior_samples.h5"
        missing_post, missing_key = load_posterior_array(missing_path)
        full_post, full_key = load_posterior_array(full_path)

        if missing_key != full_key:
            raise ValueError(f"Posterior keys differ: missing={missing_key}, full={full_key}.")
        validate_matched_posterior_shapes(missing_post, full_post)

        num_eval, num_posterior_samples, theta_dim = missing_post.shape
        observation_indices = subsample_indices(num_eval, max_observations, rng)
        sample_count = min(max_samples_per_observation, num_posterior_samples, full_post.shape[1])
        mmd_values = []
        c2st_values = []

        for observation_index in observation_indices:
            missing_samples = subsample_rows(missing_post[observation_index], sample_count, rng)
            full_samples = subsample_rows(full_post[observation_index], sample_count, rng)
            mmd_values.append(rbf_mmd2(full_samples, missing_samples))
            c2st_values.append(c2st_accuracy(full_samples, missing_samples, test_size=c2st_test_size, seed=seed))

        base_row.update(
            {
                "posterior_key": missing_key,
                "theta_dim": int(theta_dim),
                "num_eval_total": int(num_eval),
                "num_observations_used": int(observation_indices.size),
                "num_posterior_samples_total": int(num_posterior_samples),
                "num_samples_per_observation_used": int(sample_count),
                "mmd2_rbf_obs_mean": mean_values(mmd_values),
                "mmd2_rbf_obs_std": std_values(mmd_values),
                "mmd2_rbf_obs_median": median_values(mmd_values),
                "c2st_accuracy_obs_mean": mean_values(c2st_values),
                "c2st_accuracy_obs_std": std_values(c2st_values),
                "c2st_accuracy_obs_median": median_values(c2st_values),
                "status": "ok",
                "error": "",
            }
        )
    except Exception as exc:
        base_row["error"] = str(exc)

    return base_row


def load_posterior_array(path: Path) -> tuple[np.ndarray, str]:
    with h5py.File(path, "r") as h5:
        key = select_posterior_key(h5)
        samples = np.asarray(h5[key][...], dtype=np.float64)

    if samples.ndim != 3:
        raise ValueError(f"{path}::{key} must have shape (num_eval, num_samples, theta_dim).")
    return samples, key


def validate_matched_posterior_shapes(missing_post: np.ndarray, full_post: np.ndarray) -> None:
    if missing_post.ndim != 3 or full_post.ndim != 3:
        raise ValueError("Posterior arrays must have shape (num_eval, num_samples, theta_dim).")
    if missing_post.shape[0] != full_post.shape[0]:
        raise ValueError(f"num_eval differs: missing={missing_post.shape[0]}, full={full_post.shape[0]}.")
    if missing_post.shape[2] != full_post.shape[2]:
        raise ValueError(f"theta_dim differs: missing={missing_post.shape[2]}, full={full_post.shape[2]}.")


def subsample_indices(total: int, max_count: int, rng: np.random.Generator) -> np.ndarray:
    if total <= max_count:
        return np.arange(total)
    return np.sort(rng.choice(total, size=max_count, replace=False))


def build_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["problem"], row["missingness"], row["epsilon"])].append(row)

    summary_rows = []
    for key, group in sorted(groups.items()):
        ok_group = [row for row in group if row["status"] == "ok"]
        seeds = {row["seed"] for row in group}
        summary_rows.append(
            {
                "method": key[0],
                "problem": key[1],
                "missingness": key[2],
                "epsilon": key[3],
                "num_rows": len(group),
                "num_seeds": len(seeds),
                "mmd2_rbf_obs_mean_mean": mean_metric(ok_group, "mmd2_rbf_obs_mean"),
                "mmd2_rbf_obs_mean_std": std_metric(ok_group, "mmd2_rbf_obs_mean"),
                "mmd2_rbf_obs_median_mean": mean_metric(ok_group, "mmd2_rbf_obs_median"),
                "mmd2_rbf_obs_median_std": std_metric(ok_group, "mmd2_rbf_obs_median"),
                "c2st_accuracy_obs_mean_mean": mean_metric(ok_group, "c2st_accuracy_obs_mean"),
                "c2st_accuracy_obs_mean_std": std_metric(ok_group, "c2st_accuracy_obs_mean"),
                "c2st_accuracy_obs_median_mean": mean_metric(ok_group, "c2st_accuracy_obs_median"),
                "c2st_accuracy_obs_median_std": std_metric(ok_group, "c2st_accuracy_obs_median"),
            }
        )
    return summary_rows


def mean_values(values: list[float]) -> float:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    return float("nan") if not finite_values else float(statistics.fmean(finite_values))


def std_values(values: list[float]) -> float:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return float("nan")
    if len(finite_values) == 1:
        return 0.0
    return float(statistics.stdev(finite_values))


def median_values(values: list[float]) -> float:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    return float("nan") if not finite_values else float(statistics.median(finite_values))


if __name__ == "__main__":
    main()
