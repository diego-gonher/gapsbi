from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


METRICS = [
    "tarp_mae",
    "tarp_iae",
    "sbc_ks_pval_min",
    "sbc_ks_pval_mean",
    "training_time_sec",
    "posterior_sampling_time_sec",
    "diagnostics_time_sec",
    "total_runtime_sec",
]


def parse_float(value: Any) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if text == "":
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def parse_seed(value: Any) -> int | str:
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        return text


def is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def load_full_data_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("method") != "full_data":
                continue
            parsed = dict(row)
            parsed["seed"] = parse_seed(parsed.get("seed"))
            for metric in METRICS:
                parsed[metric] = parse_float(parsed.get(metric))
            rows.append(parsed)
    return rows


def group_by_problem(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["problem"])].append(row)
    return dict(groups)


def finite_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    return [float(row[metric]) for row in rows if not is_nan(row[metric])]


def summarize_metric(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    values = finite_values(rows, metric)
    if not values:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "median": float("nan"),
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


def rank_by_tarp_iae(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (is_nan(row["tarp_iae"]), row["tarp_iae"] if not is_nan(row["tarp_iae"]) else math.inf),
    )


def format_value(value: Any) -> str:
    if is_nan(value):
        return "NaN"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def format_metric_summary(rows: list[dict[str, Any]]) -> list[str]:
    lines = []
    for metric in METRICS:
        summary = summarize_metric(rows, metric)
        lines.append(
            "  "
            f"{metric}: "
            f"mean={format_value(summary['mean'])}, "
            f"std={format_value(summary['std'])}, "
            f"median={format_value(summary['median'])}, "
            f"min={format_value(summary['min'])}, "
            f"max={format_value(summary['max'])}, "
            f"count={summary['count']}"
        )
    return lines


def build_report(rows: list[dict[str, Any]]) -> str:
    groups = group_by_problem(rows)
    lines = [
        "Full-Data Diagnostics Report",
        "",
        f"Input rows: {len(rows)}",
        f"Problems: {', '.join(sorted(groups))}",
        "",
    ]

    for problem, problem_rows in sorted(groups.items()):
        ranked = rank_by_tarp_iae(problem_rows)
        best = ranked[0] if ranked else None
        median = ranked[len(ranked) // 2] if ranked else None

        lines.extend(
            [
                f"## {problem}",
                f"Seeds: {len(problem_rows)}",
                "",
                "Full-data metric summary:",
                *format_metric_summary(problem_rows),
                "",
            ]
        )

        if best is not None:
            lines.extend(
                [
                    "Best seed by lowest tarp_iae:",
                    (
                        f"  seed={best['seed']} "
                        f"tarp_iae={format_value(best['tarp_iae'])} "
                        f"tarp_mae={format_value(best['tarp_mae'])} "
                        f"run_dir={best['run_dir']}"
                    ),
                    "",
                ]
            )

        if median is not None:
            lines.extend(
                [
                    "Median seed by tarp_iae rank:",
                    (
                        f"  seed={median['seed']} "
                        f"rank={len(ranked) // 2 + 1}/{len(ranked)} "
                        f"tarp_iae={format_value(median['tarp_iae'])} "
                        f"tarp_mae={format_value(median['tarp_mae'])} "
                        f"run_dir={median['run_dir']}"
                    ),
                    "",
                ]
            )

        lines.append("Seed run paths:")
        for row in sorted(problem_rows, key=lambda item: str(item["seed"])):
            lines.append(f"  seed={row['seed']}  {row['run_dir']}")
        lines.append("")

        lines.append("Ranking of seeds by tarp_iae:")
        for rank, row in enumerate(ranked, start=1):
            lines.append(
                f"  {rank}. seed={row['seed']} "
                f"tarp_iae={format_value(row['tarp_iae'])} "
                f"tarp_mae={format_value(row['tarp_mae'])} "
                f"sbc_ks_pval_min={format_value(row['sbc_ks_pval_min'])} "
                f"total_runtime_sec={format_value(row['total_runtime_sec'])} "
                f"run_dir={row['run_dir']}"
            )
        lines.extend(["", ""])

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report seed diagnostics for Campaign 1 full-data runs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/campaign1_master_results.csv"),
        help="Path to campaign1_master_results.csv.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/analysis/full_data_diagnostics_report.txt"),
        help="Path for the full-data diagnostics report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_full_data_rows(args.input)
    report = build_report(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")

    print("Full-data diagnostics report")
    print(f"Rows: {len(rows)}")
    print(f"Problems: {len(group_by_problem(rows))}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
