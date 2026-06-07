from __future__ import annotations

import argparse
import csv
import itertools
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from compute_campaign1_shift_metrics import (
    c2st_accuracy,
    load_master_csv,
    load_posterior_samples,
    mean_metric,
    rbf_mmd2,
    std_metric,
    subsample_rows,
    write_csv,
)


SEED_PAIR_COLUMNS = [
    "problem",
    "seed_a",
    "seed_b",
    "run_dir_a",
    "run_dir_b",
    "posterior_key",
    "theta_dim",
    "num_samples_total_a",
    "num_samples_total_b",
    "num_samples_used",
    "mmd2_rbf",
    "c2st_accuracy",
    "status",
    "error",
]

SUMMARY_COLUMNS = [
    "problem",
    "num_pairs",
    "mmd2_rbf_mean",
    "mmd2_rbf_std",
    "c2st_accuracy_mean",
    "c2st_accuracy_std",
]


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    rows = load_master_csv(args.input)
    full_data_rows = [row for row in rows if row.get("method") == "full_data"]
    seed_pairs = build_full_data_seed_pairs(full_data_rows)

    print("Full-data seed posterior shift metrics")
    print(f"Input CSV: {args.input}")
    print(f"Master CSV rows: {len(rows)}")
    print(f"Full-data runs: {len(full_data_rows)}")
    print(f"Seed pairs to process: {len(seed_pairs)}")
    print(f"Max samples: {args.max_samples}")
    print(f"Seed: {args.seed}")
    print(f"Seed-pair output: {args.out}")
    print(f"Summary output: {args.summary_out}")

    pair_rows = build_seed_pair_metric_rows(
        seed_pairs,
        max_samples=args.max_samples,
        seed=args.seed,
        c2st_test_size=args.c2st_test_size,
    )
    summary_rows = build_summary_rows(pair_rows)

    write_csv(args.out, pair_rows, SEED_PAIR_COLUMNS)
    write_csv(args.summary_out, summary_rows, SUMMARY_COLUMNS)

    ok_rows = sum(row["status"] == "ok" for row in pair_rows)
    failed_rows = len(pair_rows) - ok_rows
    total_runtime_sec = time.perf_counter() - start_time
    print("Full-data seed posterior shift metrics complete")
    print(f"Successful rows: {ok_rows}")
    print(f"Failed rows: {failed_rows}")
    print(f"Total runtime: {total_runtime_sec:.2f} sec")
    print(f"Seed-pair output: {args.out}")
    print(f"Summary output: {args.summary_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute posterior shift metrics between full-data NPE seeds.",
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
        default=Path("outputs/analysis/full_data_seed_shift_metrics.csv"),
        help="Path for seed-pair shift metrics.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("outputs/analysis/full_data_seed_shift_metrics_summary.csv"),
        help="Path for grouped seed-pair summaries.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2_000,
        help="Maximum flattened posterior samples to use per distribution.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for subsampling and C2ST split.")
    parser.add_argument("--c2st-test-size", type=float, default=0.5, help="Held-out fraction for C2ST.")
    return parser.parse_args()


def build_full_data_seed_pairs(rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rows_by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_problem[str(row["problem"])].append(row)

    pairs = []
    for problem in sorted(rows_by_problem):
        problem_rows = sorted(rows_by_problem[problem], key=lambda row: int(row["seed"]))
        pairs.extend(itertools.combinations(problem_rows, 2))
    return pairs


def build_seed_pair_metric_rows(
    seed_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    max_samples: int,
    seed: int,
    c2st_test_size: float,
) -> list[dict[str, Any]]:
    if max_samples < 2:
        raise ValueError("--max-samples must be at least 2.")

    pair_rows = []
    progress = tqdm(seed_pairs, desc="Computing full-data seed shifts", total=len(seed_pairs))
    for index, (row_a, row_b) in enumerate(progress):
        progress.set_postfix(problem=row_a.get("problem", ""), seed_a=row_a.get("seed", ""), seed_b=row_b.get("seed", ""))
        rng = np.random.default_rng(seed + index)
        row = build_seed_pair_metric_row(
            row_a,
            row_b,
            max_samples=max_samples,
            rng=rng,
            c2st_test_size=c2st_test_size,
            seed=seed,
        )
        if row["status"] == "error":
            tqdm.write(
                "Error computing full-data seed shift for "
                f"{row['problem']} seed_a={row['seed_a']} seed_b={row['seed_b']}: {row['error']}"
            )
        pair_rows.append(row)
    return pair_rows


def build_seed_pair_metric_row(
    row_a: dict[str, Any],
    row_b: dict[str, Any],
    max_samples: int,
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
        "num_samples_total_a": "",
        "num_samples_total_b": "",
        "num_samples_used": "",
        "mmd2_rbf": "",
        "c2st_accuracy": "",
        "status": "error",
        "error": "",
    }

    try:
        posterior_path_a = Path(str(row_a["run_dir"])) / "posterior_samples.h5"
        posterior_path_b = Path(str(row_b["run_dir"])) / "posterior_samples.h5"
        samples_a, key_a = load_posterior_samples(posterior_path_a)
        samples_b, key_b = load_posterior_samples(posterior_path_b)

        if key_a != key_b:
            raise ValueError(f"Posterior keys differ: seed_a={key_a}, seed_b={key_b}.")
        if samples_a.shape[1] != samples_b.shape[1]:
            raise ValueError(f"Theta dimensions differ: seed_a={samples_a.shape[1]}, seed_b={samples_b.shape[1]}.")

        total_a = int(samples_a.shape[0])
        total_b = int(samples_b.shape[0])
        sample_count = min(max_samples, total_a, total_b)
        subsample_a = subsample_rows(samples_a, sample_count, rng)
        subsample_b = subsample_rows(samples_b, sample_count, rng)

        base_row.update(
            {
                "posterior_key": key_a,
                "theta_dim": int(samples_a.shape[1]),
                "num_samples_total_a": total_a,
                "num_samples_total_b": total_b,
                "num_samples_used": sample_count,
                "mmd2_rbf": rbf_mmd2(subsample_a, subsample_b),
                "c2st_accuracy": c2st_accuracy(
                    subsample_a,
                    subsample_b,
                    test_size=c2st_test_size,
                    seed=seed,
                ),
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
                "mmd2_rbf_mean": mean_metric(ok_group, "mmd2_rbf"),
                "mmd2_rbf_std": std_metric(ok_group, "mmd2_rbf"),
                "c2st_accuracy_mean": mean_metric(ok_group, "c2st_accuracy"),
                "c2st_accuracy_std": std_metric(ok_group, "c2st_accuracy"),
            }
        )
    return summary_rows


if __name__ == "__main__":
    main()
