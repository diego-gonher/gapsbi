from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


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
    "num_missing_samples_total",
    "num_full_samples_total",
    "num_samples_used",
    "mmd2_rbf",
    "c2st_accuracy",
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
    "mmd2_rbf_mean",
    "mmd2_rbf_std",
    "c2st_accuracy_mean",
    "c2st_accuracy_std",
]


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    rows = load_master_csv(args.input)
    missing_rows = [row for row in rows if row.get("method") != "full_data"]

    print("Campaign 1 posterior shift metrics")
    print(f"Input CSV: {args.input}")
    print(f"Master CSV rows: {len(rows)}")
    print(f"Missing-data runs to process: {len(missing_rows)}")
    print(f"Max samples: {args.max_samples}")
    print(f"Seed: {args.seed}")
    print(f"Seed-level output: {args.out}")
    print(f"Summary output: {args.summary_out}")

    shift_rows = build_shift_metric_rows(
        rows,
        max_samples=args.max_samples,
        seed=args.seed,
        c2st_test_size=args.c2st_test_size,
    )
    summary_rows = build_summary_rows(shift_rows)

    write_csv(args.out, shift_rows, SEED_LEVEL_COLUMNS)
    write_csv(args.summary_out, summary_rows, SUMMARY_COLUMNS)

    ok_rows = sum(row["status"] == "ok" for row in shift_rows)
    failed_rows = len(shift_rows) - ok_rows
    total_runtime_sec = time.perf_counter() - start_time
    print("Campaign 1 posterior shift metrics complete")
    print(f"Successful rows: {ok_rows}")
    print(f"Failed rows: {failed_rows}")
    print(f"Total runtime: {total_runtime_sec:.2f} sec")
    print(f"Seed-level output: {args.out}")
    print(f"Summary output: {args.summary_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute MMD and C2ST posterior shift metrics for Campaign 1.",
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
        default=Path("outputs/analysis/campaign1_shift_metrics.csv"),
        help="Path for seed-level shift metrics.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("outputs/analysis/campaign1_shift_metrics_summary.csv"),
        help="Path for grouped shift metric summaries.",
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


def load_master_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def build_shift_metric_rows(
    rows: list[dict[str, Any]],
    max_samples: int,
    seed: int,
    c2st_test_size: float,
) -> list[dict[str, Any]]:
    if max_samples < 2:
        raise ValueError("--max-samples must be at least 2.")

    full_rows = {
        (row["problem"], row["seed"]): row
        for row in rows
        if row.get("method") == "full_data"
    }
    missing_rows = [row for row in rows if row.get("method") != "full_data"]

    shift_rows = []
    progress = tqdm(missing_rows, desc="Computing shift metrics", total=len(missing_rows))
    for index, missing_row in enumerate(progress):
        progress.set_postfix(
            problem=missing_row.get("problem", ""),
            method=missing_row.get("method", ""),
            missingness=missing_row.get("missingness", ""),
            epsilon=missing_row.get("epsilon", ""),
        )
        rng = np.random.default_rng(seed + index)
        row = build_shift_metric_row(
            missing_row,
            full_rows,
            max_samples=max_samples,
            rng=rng,
            c2st_test_size=c2st_test_size,
            seed=seed,
        )
        if row["status"] == "error":
            tqdm.write(
                "Error computing shift metrics for "
                f"{row['method']} {row['problem']} {row['missingness']} "
                f"eps={row['epsilon']} seed={row['seed']}: {row['error']}"
            )
        shift_rows.append(row)
    return shift_rows


def build_shift_metric_row(
    missing_row: dict[str, Any],
    full_rows: dict[tuple[str, str], dict[str, Any]],
    max_samples: int,
    rng: np.random.Generator,
    c2st_test_size: float,
    seed: int,
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
        "num_missing_samples_total": "",
        "num_full_samples_total": "",
        "num_samples_used": "",
        "mmd2_rbf": "",
        "c2st_accuracy": "",
        "status": "error",
        "error": "",
    }

    try:
        full_row = full_rows[(str(missing_row["problem"]), str(missing_row["seed"]))]
        base_row["full_run_dir"] = full_row.get("run_dir", "")

        missing_path = Path(str(missing_row["run_dir"])) / "posterior_samples.h5"
        full_path = Path(str(full_row["run_dir"])) / "posterior_samples.h5"
        missing_samples, missing_key = load_posterior_samples(missing_path)
        full_samples, full_key = load_posterior_samples(full_path)

        if missing_key != full_key:
            raise ValueError(f"Posterior keys differ: missing={missing_key}, full={full_key}.")
        if missing_samples.shape[1] != full_samples.shape[1]:
            raise ValueError(
                f"Theta dimensions differ: missing={missing_samples.shape[1]}, full={full_samples.shape[1]}."
            )

        missing_total = int(missing_samples.shape[0])
        full_total = int(full_samples.shape[0])
        sample_count = min(max_samples, missing_total, full_total)
        missing_subsample = subsample_rows(missing_samples, sample_count, rng)
        full_subsample = subsample_rows(full_samples, sample_count, rng)

        base_row.update(
            {
                "posterior_key": missing_key,
                "theta_dim": int(missing_samples.shape[1]),
                "num_missing_samples_total": missing_total,
                "num_full_samples_total": full_total,
                "num_samples_used": sample_count,
                "mmd2_rbf": rbf_mmd2(full_subsample, missing_subsample),
                "c2st_accuracy": c2st_accuracy(
                    full_subsample,
                    missing_subsample,
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


def load_posterior_samples(path: Path) -> tuple[np.ndarray, str]:
    with h5py.File(path, "r") as h5:
        key = select_posterior_key(h5)
        samples = np.asarray(h5[key][...], dtype=np.float64)

    if samples.ndim != 3:
        raise ValueError(f"{path}::{key} must have shape (num_eval, num_samples, theta_dim).")
    num_eval, num_posterior_samples, theta_dim = samples.shape
    return samples.reshape(num_eval * num_posterior_samples, theta_dim), key


def select_posterior_key(h5: h5py.File) -> str:
    if "theta_posterior_scaled" in h5:
        return "theta_posterior_scaled"
    if "theta_posterior" in h5:
        return "theta_posterior"
    raise KeyError("Missing theta_posterior_scaled and theta_posterior datasets.")


def subsample_rows(samples: np.ndarray, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    if samples.shape[0] <= max_samples:
        return samples
    indices = rng.choice(samples.shape[0], size=max_samples, replace=False)
    return samples[indices]


def rbf_mmd2(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must be 2D arrays with matching feature dimensions.")

    gamma = median_heuristic_gamma(x, y)
    kxx = np.exp(-gamma * squared_distances(x, x)).mean()
    kyy = np.exp(-gamma * squared_distances(y, y)).mean()
    kxy = np.exp(-gamma * squared_distances(x, y)).mean()
    return max(0.0, float(kxx + kyy - 2.0 * kxy))


def median_heuristic_gamma(x: np.ndarray, y: np.ndarray) -> float:
    pooled = np.vstack([x, y])
    distances = squared_distances(pooled, pooled)
    positive_distances = distances[distances > 0.0]
    if positive_distances.size == 0:
        return 1.0

    median_sqdist = float(np.median(positive_distances))
    if not math.isfinite(median_sqdist) or median_sqdist <= 0.0:
        return 1.0
    return 1.0 / (2.0 * median_sqdist)


def squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_norm = np.sum(x * x, axis=1)[:, None]
    y_norm = np.sum(y * y, axis=1)[None, :]
    return np.maximum(x_norm + y_norm - 2.0 * (x @ y.T), 0.0)


def c2st_accuracy(x: np.ndarray, y: np.ndarray, test_size: float = 0.5, seed: int = 0) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    features = np.vstack([x, y])
    labels = np.concatenate([np.zeros(x.shape[0], dtype=int), np.ones(y.shape[0], dtype=int)])

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1_000, random_state=seed),
    )
    classifier.fit(x_train, y_train)
    return float(classifier.score(x_test, y_test))


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
                "mmd2_rbf_mean": mean_metric(ok_group, "mmd2_rbf"),
                "mmd2_rbf_std": std_metric(ok_group, "mmd2_rbf"),
                "c2st_accuracy_mean": mean_metric(ok_group, "c2st_accuracy"),
                "c2st_accuracy_std": std_metric(ok_group, "c2st_accuracy"),
            }
        )
    return summary_rows


def mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = finite_metric_values(rows, key)
    return float("nan") if not values else float(statistics.fmean(values))


def std_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = finite_metric_values(rows, key)
    if not values:
        return float("nan")
    if len(values) == 1:
        return 0.0
    return float(statistics.stdev(values))


def finite_metric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        try:
            value = float(row[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
