from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPECTED_ROWS = 560
EXPECTED_CONFIGS = 112
EXPECTED_SEEDS_PER_CONFIG = 5

CONFIG_COLUMNS = ["method", "problem", "missingness", "epsilon", "experiment"]
METHOD_GROUP_COLUMNS = ["method", "problem", "missingness", "epsilon"]
KEY_METRICS = [
    "tarp_mae",
    "tarp_iae",
    "sbc_ks_pval_min",
    "sbc_ks_pval_mean",
    "training_time_sec",
    "posterior_sampling_time_sec",
    "diagnostics_time_sec",
    "total_runtime_sec",
]
SUMMARY_STATS = ["mean", "std", "min", "max", "median", "count"]


def _nan() -> float:
    return float("nan")


def parse_float(value: Any) -> float:
    if value is None:
        return _nan()
    text = str(value).strip()
    if text == "":
        return _nan()
    try:
        return float(text)
    except ValueError:
        return _nan()


def parse_seed(value: Any) -> int | str:
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        return text


def is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def is_ok_status(status: Any) -> bool:
    return str(status).strip().lower() in {"ok", "success", "completed"}


def load_master_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = dict(row)
            parsed["seed"] = parse_seed(parsed.get("seed"))
            for metric in KEY_METRICS:
                parsed[metric] = parse_float(parsed.get(metric))
            parsed["epsilon"] = parse_float(parsed.get("epsilon"))
            rows.append(parsed)
    return rows


def config_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[column] for column in CONFIG_COLUMNS)


def method_group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[column] for column in METHOD_GROUP_COLUMNS)


def finite_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    values = [row[metric] for row in rows]
    return [float(value) for value in values if not is_nan(value)]


def summarize_metric(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "mean": _nan(),
            "std": _nan(),
            "min": _nan(),
            "max": _nan(),
            "median": _nan(),
            "count": 0,
        }

    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
        "count": len(values),
    }


def grouped_rows(
    rows: list[dict[str, Any]],
    group_columns: list[str],
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[column] for column in group_columns)].append(row)
    return dict(groups)


def build_summary_rows(
    rows: list[dict[str, Any]],
    group_columns: list[str],
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped_rows(rows, group_columns).items()):
        summary: dict[str, Any] = dict(zip(group_columns, key, strict=True))
        summary["num_rows"] = len(group)
        summary["num_seeds"] = len({row["seed"] for row in group})
        for metric in KEY_METRICS:
            stats = summarize_metric(finite_values(group, metric))
            for stat_name in SUMMARY_STATS:
                summary[f"{metric}_{stat_name}"] = stats[stat_name]
        summary_rows.append(summary)
    return summary_rows


def summary_fieldnames(group_columns: list[str]) -> list[str]:
    fields = [*group_columns, "num_rows", "num_seeds"]
    for metric in KEY_METRICS:
        fields.extend(f"{metric}_{stat_name}" for stat_name in SUMMARY_STATS)
    return fields


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def status_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(row.get("status", "ok")) for row in rows)


def nan_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        metric: sum(1 for row in rows if is_nan(row[metric]))
        for metric in KEY_METRICS
    }


def seed_count_issues(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for key, group in sorted(grouped_rows(rows, CONFIG_COLUMNS).items()):
        seeds = sorted({row["seed"] for row in group})
        if len(seeds) != EXPECTED_SEEDS_PER_CONFIG:
            issues.append(
                {
                    **dict(zip(CONFIG_COLUMNS, key, strict=True)),
                    "num_seeds": len(seeds),
                    "seeds": ",".join(str(seed) for seed in seeds),
                }
            )
    return issues


def failed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if not is_ok_status(row.get("status", "ok"))]


def _sort_key_high(row: dict[str, Any], column: str) -> tuple[bool, float]:
    value = row.get(column, _nan())
    return (not is_nan(value), float(value) if not is_nan(value) else -math.inf)


def _sort_key_low(row: dict[str, Any], column: str) -> tuple[bool, float]:
    value = row.get(column, _nan())
    return (not is_nan(value), -float(value) if not is_nan(value) else -math.inf)


def top_high(rows: list[dict[str, Any]], column: str, n: int = 10) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: _sort_key_high(row, column), reverse=True)[:n]


def top_low(rows: list[dict[str, Any]], column: str, n: int = 10) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: _sort_key_low(row, column), reverse=True)[:n]


def format_config(row: dict[str, Any]) -> str:
    return (
        f"{row['method']} | {row['problem']} | {row['missingness']} | "
        f"eps={float(row['epsilon']):.2f} | {row['experiment']}"
    )


def format_top_table(title: str, rows: list[dict[str, Any]], metric_column: str) -> list[str]:
    lines = [title]
    if not rows:
        return [*lines, "  None"]
    for idx, row in enumerate(rows, start=1):
        value = row.get(metric_column, _nan())
        value_text = "NaN" if is_nan(value) else f"{float(value):.6g}"
        lines.append(f"  {idx:2d}. {value_text}  {format_config(row)}")
    return lines


def build_sanity_report(
    rows: list[dict[str, Any]],
    config_summary_rows: list[dict[str, Any]],
) -> str:
    unique_configs = {config_key(row) for row in rows}
    unique_seeds = {row["seed"] for row in rows}
    statuses = status_counts(rows)
    seed_issues = seed_count_issues(rows)
    failures = failed_rows(rows)
    nans = nan_counts(rows)

    lines = [
        "Campaign 1 Stability Sanity Report",
        "",
        f"Total rows: {len(rows)} (expected {EXPECTED_ROWS})",
        f"Unique configs: {len(unique_configs)} (expected {EXPECTED_CONFIGS})",
        f"Unique seeds: {len(unique_seeds)}",
        f"Expected seeds per config: {EXPECTED_SEEDS_PER_CONFIG}",
        "",
        "Status counts:",
    ]
    for status, count in sorted(statuses.items()):
        lines.append(f"  {status}: {count}")

    lines.extend(["", f"Failed/non-ok runs: {len(failures)}"])
    if failures:
        for row in failures[:20]:
            lines.append(f"  {row.get('status')}  {format_config(row)} seed={row.get('seed')}")
        if len(failures) > 20:
            lines.append(f"  ... {len(failures) - 20} more")

    lines.extend(["", f"Configs with fewer/more than {EXPECTED_SEEDS_PER_CONFIG} seeds: {len(seed_issues)}"])
    if seed_issues:
        for issue in seed_issues:
            lines.append(
                f"  {format_config(issue)} num_seeds={issue['num_seeds']} seeds=[{issue['seeds']}]"
            )

    lines.extend(["", "NaN counts for key metrics:"])
    for metric in KEY_METRICS:
        lines.append(f"  {metric}: {nans[metric]}")

    lines.extend(["", *format_top_table("Top 10 worst configs by mean tarp_iae:", top_high(config_summary_rows, "tarp_iae_mean"), "tarp_iae_mean")])
    lines.extend(["", *format_top_table("Top 10 worst configs by mean tarp_mae:", top_high(config_summary_rows, "tarp_mae_mean"), "tarp_mae_mean")])
    lines.extend(
        [
            "",
            *format_top_table(
                "Top 10 worst configs by lowest mean sbc_ks_pval_min:",
                top_low(config_summary_rows, "sbc_ks_pval_min_mean"),
                "sbc_ks_pval_min_mean",
            ),
        ]
    )
    lines.extend(
        [
            "",
            *format_top_table(
                "Top 10 slowest configs by mean total_runtime_sec:",
                top_high(config_summary_rows, "total_runtime_sec_mean"),
                "total_runtime_sec_mean",
            ),
        ]
    )

    return "\n".join(lines) + "\n"


def write_outputs(rows: list[dict[str, Any]], outdir: Path) -> tuple[Path, Path, Path]:
    config_summary = build_summary_rows(rows, CONFIG_COLUMNS)
    method_summary = build_summary_rows(rows, METHOD_GROUP_COLUMNS)

    config_summary_path = outdir / "campaign1_config_summary.csv"
    method_summary_path = outdir / "campaign1_method_summary.csv"
    report_path = outdir / "campaign1_sanity_report.txt"

    write_csv(config_summary_path, config_summary, summary_fieldnames(CONFIG_COLUMNS))
    write_csv(method_summary_path, method_summary, summary_fieldnames(METHOD_GROUP_COLUMNS))
    outdir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_sanity_report(rows, config_summary), encoding="utf-8")

    return config_summary_path, method_summary_path, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Campaign 1 seed-level results for stability and sanity checks.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/campaign1_master_results.csv"),
        help="Path to campaign1_master_results.csv.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/analysis"),
        help="Directory for analysis CSVs and sanity report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_master_csv(args.input)
    config_summary_path, method_summary_path, report_path = write_outputs(rows, args.outdir)

    print("Campaign 1 stability analysis summary")
    print(f"Rows: {len(rows)}")
    print(f"Unique configs: {len({config_key(row) for row in rows})}")
    print(f"Unique seeds: {len({row['seed'] for row in rows})}")
    print(f"Failed/non-ok runs: {len(failed_rows(rows))}")
    print(f"Config summary: {config_summary_path}")
    print(f"Method summary: {method_summary_path}")
    print(f"Sanity report: {report_path}")


if __name__ == "__main__":
    main()
