from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from compute_campaign1_per_observation_shift_metrics import (
    c2st_accuracy,
    load_master_csv,
    load_posterior_array,
    mean_metric,
    mean_values,
    median_values,
    rbf_mmd2,
    std_metric,
    std_values,
    subsample_indices,
    subsample_rows,
    validate_matched_posterior_shapes,
    write_csv,
)
from compute_full_data_seed_shift_metrics import build_full_data_seed_pairs


SEED_PAIR_COLUMNS = [
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
    "problem",
    "num_pairs",
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
    full_data_rows = [row for row in rows if row.get("method") == "full_data"]
    seed_pairs = build_full_data_seed_pairs(full_data_rows)

    print("Full-data per-observation seed posterior shift metrics")
    print(f"Input CSV: {args.input}")
    print(f"Master CSV rows: {len(rows)}")
    print(f"Full-data runs: {len(full_data_rows)}")
    print(f"Seed pairs to process: {len(seed_pairs)}")
    print(f"Max observations: {args.max_observations}")
    print(f"Max samples per observation: {args.max_samples_per_observation}")
    print(f"Seed: {args.seed}")
    print(f"Seed-pair output: {args.out}")
    print(f"Summary output: {args.summary_out}")

    pair_rows = build_seed_pair_metric_rows(
        seed_pairs,
        max_observations=args.max_observations,
        max_samples_per_observation=args.max_samples_per_observation,
        seed=args.seed,
        c2st_test_size=args.c2st_test_size,
    )
    summary_rows = build_summary_rows(pair_rows)

    write_csv(args.out, pair_rows, SEED_PAIR_COLUMNS)
    write_csv(args.summary_out, summary_rows, SUMMARY_COLUMNS)

    ok_rows = sum(row["status"] == "ok" for row in pair_rows)
    failed_rows = len(pair_rows) - ok_rows
    total_runtime_sec = time.perf_counter() - start_time
    print("Full-data per-observation seed posterior shift metrics complete")
    print(f"Successful rows: {ok_rows}")
    print(f"Failed rows: {failed_rows}")
    print(f"Total runtime: {total_runtime_sec:.2f} sec")
    print(f"Seed-pair output: {args.out}")
    print(f"Summary output: {args.summary_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-observation posterior shift metrics between full-data NPE seeds.",
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
        default=Path("outputs/analysis/full_data_per_observation_seed_shift_metrics.csv"),
        help="Path for seed-pair per-observation shift metrics.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("outputs/analysis/full_data_per_observation_seed_shift_metrics_summary.csv"),
        help="Path for grouped seed-pair summaries.",
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
    parser.add_argument("--seed", type=int, default=0, help="Random seed for subsampling and C2ST splits.")
    parser.add_argument("--c2st-test-size", type=float, default=0.5, help="Held-out fraction for C2ST.")
    return parser.parse_args()


def build_seed_pair_metric_rows(
    seed_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    max_observations: int,
    max_samples_per_observation: int,
    seed: int,
    c2st_test_size: float,
) -> list[dict[str, Any]]:
    if max_observations < 1:
        raise ValueError("--max-observations must be at least 1.")
    if max_samples_per_observation < 2:
        raise ValueError("--max-samples-per-observation must be at least 2.")

    pair_rows = []
    progress = tqdm(seed_pairs, desc="Computing full-data per-observation shifts", total=len(seed_pairs))
    for index, (row_a, row_b) in enumerate(progress):
        progress.set_postfix(problem=row_a.get("problem", ""), seed_a=row_a.get("seed", ""), seed_b=row_b.get("seed", ""))
        rng = np.random.default_rng(seed + index)
        row = build_seed_pair_metric_row(
            row_a,
            row_b,
            max_observations=max_observations,
            max_samples_per_observation=max_samples_per_observation,
            rng=rng,
            c2st_test_size=c2st_test_size,
            seed=seed,
        )
        if row["status"] == "error":
            tqdm.write(
                "Error computing full-data per-observation seed shift for "
                f"{row['problem']} seed_a={row['seed_a']} seed_b={row['seed_b']}: {row['error']}"
            )
        pair_rows.append(row)
    return pair_rows


def build_seed_pair_metric_row(
    row_a: dict[str, Any],
    row_b: dict[str, Any],
    max_observations: int,
    max_samples_per_observation: int,
    rng: np.random.Generator,
    c2st_test_size: float,
    seed: int,
) -> dict[str, Any]:
    base_row = {
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
        "mmd2_rbf_obs_mean": "",
        "mmd2_rbf_obs_std": "",
        "mmd2_rbf_obs_median": "",
        "c2st_accuracy_obs_mean": "",
        "c2st_accuracy_obs_std": "",
        "c2st_accuracy_obs_median": "",
        "status": "error",
        "error": "",
    }

    try:
        posterior_path_a = Path(str(row_a["run_dir"])) / "posterior_samples.h5"
        posterior_path_b = Path(str(row_b["run_dir"])) / "posterior_samples.h5"
        posterior_a, key_a = load_posterior_array(posterior_path_a)
        posterior_b, key_b = load_posterior_array(posterior_path_b)

        if key_a != key_b:
            raise ValueError(f"Posterior keys differ: seed_a={key_a}, seed_b={key_b}.")
        validate_matched_posterior_shapes(posterior_a, posterior_b)

        num_eval, num_posterior_samples, theta_dim = posterior_a.shape
        observation_indices = subsample_indices(num_eval, max_observations, rng)
        sample_count = min(max_samples_per_observation, num_posterior_samples, posterior_b.shape[1])
        mmd_values = []
        c2st_values = []

        for observation_index in observation_indices:
            samples_a = subsample_rows(posterior_a[observation_index], sample_count, rng)
            samples_b = subsample_rows(posterior_b[observation_index], sample_count, rng)
            mmd_values.append(rbf_mmd2(samples_a, samples_b))
            c2st_values.append(c2st_accuracy(samples_a, samples_b, test_size=c2st_test_size, seed=seed))

        base_row.update(
            {
                "posterior_key": key_a,
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


def build_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_problem[str(row["problem"])].append(row)

    summary_rows = []
    for problem, group in sorted(rows_by_problem.items()):
        ok_group = [row for row in group if row["status"] == "ok"]
        summary_rows.append(
            {
                "problem": problem,
                "num_pairs": len(group),
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


if __name__ == "__main__":
    main()
