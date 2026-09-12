from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT_PATH = Path("outputs/analysis/full_experiment_results_with_diagnostics.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/analysis/aggregates")
GROUPINGS = {
    "aggregate_by_config.csv": ["budget", "method", "problem", "missingness", "epsilon"],
    "aggregate_by_method.csv": ["budget", "method"],
    "aggregate_by_method_problem.csv": ["budget", "method", "problem"],
    "aggregate_by_method_missingness.csv": ["budget", "method", "missingness"],
    "aggregate_by_method_epsilon.csv": ["budget", "method", "epsilon"],
    "aggregate_by_budget.csv": ["budget"],
}
SCIENTIFIC_METRICS = [
    "tarp_mae",
    "tarp_iae",
    "tarp_atc",
    "sbc_ks_pval_min",
    "sbc_ks_pval_mean",
    "reference_c2st_accuracy_mean",
    "reference_c2st_accuracy_median",
    "reference_c2st_accuracy_max",
    "reference_posterior_mean_shift_mean",
    "reference_posterior_mean_shift_median",
    "reference_posterior_mean_shift_max",
    "reference_covariance_trace_ratio_mean",
    "reference_covariance_trace_ratio_median",
    "reference_covariance_trace_ratio_max",
]
RUNTIME_METRICS = [
    "training_time_sec",
    "posterior_sampling_time_sec",
    "diagnostics_time_sec",
    "total_runtime_sec",
    "epochs_trained",
    "best_validation_loss",
    "final_train_loss",
    "final_validation_loss",
    "best_val_loss",
]
P_VALUE_METRICS = {
    "tarp_ks_pvalue",
    "sbc_ks_pval_min",
    "sbc_ks_pval_mean",
}
STANDARD_STATS = ("mean", "std", "se", "median", "min", "max", "count")
P_VALUE_STATS = (
    "mean",
    "std",
    "se",
    "median",
    "min",
    "max",
    "frac_below_0_05",
    "frac_below_0_01",
    "count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate enriched GapSBI experiment results across seeds.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input enriched results CSV. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for aggregate CSVs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include rows whose compile/diagnostics status is not ok.",
    )
    return parser.parse_args()


def _nan() -> float:
    return float("nan")


def parse_float(value: Any) -> float:
    if value is None:
        return _nan()
    text = str(value).strip()
    if not text:
        return _nan()
    try:
        return float(text)
    except ValueError:
        return _nan()


def is_finite(value: float) -> bool:
    return math.isfinite(value)


def is_ok_row(row: dict[str, str]) -> bool:
    compile_status = str(row.get("compile_status", "ok")).strip().lower()
    diagnostics_status = str(row.get("diagnostics_status", "ok")).strip().lower()
    return compile_status == "ok" and diagnostics_status == "ok"


def parse_seed(value: Any) -> int | str:
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        return text


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def available_metrics(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    fieldnames = set().union(*(row.keys() for row in rows))
    return [
        metric
        for metric in [*SCIENTIFIC_METRICS, *RUNTIME_METRICS]
        if metric in fieldnames
    ]


def finite_metric_values(rows: list[dict[str, str]], metric: str) -> list[float]:
    values = [parse_float(row.get(metric)) for row in rows]
    return [value for value in values if is_finite(value)]


def summarize_standard(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "mean": _nan(),
            "std": _nan(),
            "se": _nan(),
            "median": _nan(),
            "min": _nan(),
            "max": _nan(),
            "count": 0,
        }
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": statistics.fmean(values),
        "std": std,
        "se": std / math.sqrt(len(values)),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "count": len(values),
    }


def summarize_p_value(values: list[float]) -> dict[str, Any]:
    summary = summarize_standard(values)
    if values:
        summary["frac_below_0_05"] = sum(value < 0.05 for value in values) / len(values)
        summary["frac_below_0_01"] = sum(value < 0.01 for value in values) / len(values)
    else:
        summary["frac_below_0_05"] = _nan()
        summary["frac_below_0_01"] = _nan()
    return summary


def group_rows(
    rows: list[dict[str, str]],
    group_columns: list[str],
) -> dict[tuple[str, ...], list[dict[str, str]]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(column, "")) for column in group_columns)].append(row)
    return dict(groups)


def seed_list(rows: list[dict[str, str]]) -> str:
    seeds = sorted({parse_seed(row.get("seed", "")) for row in rows}, key=lambda x: str(x))
    return ",".join(str(seed) for seed in seeds)


def build_summary_rows(
    rows: list[dict[str, str]],
    group_columns: list[str],
    metrics: list[str],
) -> list[dict[str, Any]]:
    summaries = []
    for key, group in sorted(group_rows(rows, group_columns).items()):
        summary: dict[str, Any] = dict(zip(group_columns, key, strict=True))
        ok_count = sum(is_ok_row(row) for row in group)
        summary.update(
            {
                "num_rows": len(group),
                "num_ok": ok_count,
                "num_failed": len(group) - ok_count,
                "num_seeds": len({row.get("seed", "") for row in group}),
                "seeds": seed_list(group),
            }
        )
        for metric in metrics:
            values = finite_metric_values(group, metric)
            if metric in P_VALUE_METRICS:
                stats = summarize_p_value(values)
                stat_names = P_VALUE_STATS
            else:
                stats = summarize_standard(values)
                stat_names = STANDARD_STATS
            for stat_name in stat_names:
                summary[f"{metric}_{stat_name}"] = stats[stat_name]
        summaries.append(summary)
    return summaries


def fieldnames_for(group_columns: list[str], metrics: list[str]) -> list[str]:
    fields = [*group_columns, "num_rows", "num_ok", "num_failed", "num_seeds", "seeds"]
    for metric in metrics:
        stat_names = P_VALUE_STATS if metric in P_VALUE_METRICS else STANDARD_STATS
        fields.extend(f"{metric}_{stat_name}" for stat_name in stat_names)
    return fields


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_aggregates(
    rows: list[dict[str, str]],
    outdir: Path,
    metrics: list[str],
) -> list[Path]:
    paths = []
    for filename, group_columns in GROUPINGS.items():
        summary_rows = build_summary_rows(rows, group_columns, metrics)
        path = outdir / filename
        write_csv(path, summary_rows, fieldnames_for(group_columns, metrics))
        paths.append(path)
    return paths


def print_summary(
    input_path: Path,
    outdir: Path,
    input_rows: list[dict[str, str]],
    rows: list[dict[str, str]],
    metrics: list[str],
    paths: list[Path],
) -> None:
    print("Experiment aggregation summary")
    print(f"Input: {input_path}")
    print(f"Input rows: {len(input_rows)}")
    print(f"Aggregated rows: {len(rows)}")
    print(f"Metrics: {len(metrics)}")
    print(f"Output directory: {outdir}")
    for path in paths:
        print(f"Wrote: {path}")


def main() -> None:
    args = parse_args()
    input_rows = read_csv(args.input)
    rows = input_rows if args.include_failed else [row for row in input_rows if is_ok_row(row)]
    metrics = available_metrics(rows)
    paths = write_aggregates(rows, args.outdir, metrics)
    print_summary(args.input, args.outdir, input_rows, rows, metrics, paths)


if __name__ == "__main__":
    main()
