from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOTS = (
    Path("outputs_high_sim_budget"),
    Path("outputs_mid_sim_budget"),
    Path("outputs_low_sim_budget"),
)
DEFAULT_OUTPUT_PATH = Path("outputs/analysis/full_experiment_results.csv")
IDENTIFIER_COLUMNS = [
    "budget",
    "method",
    "problem",
    "missingness",
    "epsilon",
    "seed",
    "experiment",
    "run_dir",
    "summary_path",
    "compile_status",
    "compile_error",
]
EPSILON_RE = re.compile(r"eps(\d{3})")
SEED_RE = re.compile(r"^seed_(\d+)$")
LIST_SUMMARY_STATS = ("mean", "std", "min", "max", "median", "count")


def _nan() -> float:
    return float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile experiment seed-level summary.json files into one CSV.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        action="append",
        dest="outputs_dirs",
        help=(
            "Output root to scan. May be passed multiple times. Defaults to "
            "outputs_high_sim_budget, outputs_mid_sim_budget, and outputs_low_sim_budget."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"CSV output path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise on malformed summaries instead of writing failed compile rows.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the compile summary without writing a CSV.",
    )
    return parser.parse_args()


def budget_from_root(root: Path) -> str:
    name = root.name
    return name.removeprefix("outputs_")


def parse_epsilon_from_text(text: str) -> float:
    match = EPSILON_RE.search(text)
    if match is None:
        return _nan()
    return int(match.group(1)) / 100.0


def seed_from_path(summary_path: Path) -> int | str:
    seed_dir = summary_path.parent.name
    match = SEED_RE.match(seed_dir)
    if match is None:
        return ""
    return int(match.group(1))


def is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def is_numeric_sequence(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    return all(isinstance(item, int | float) and not isinstance(item, bool) for item in value)


def finite_numbers(value: list[Any]) -> list[float]:
    numbers = []
    for item in value:
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            numbers.append(number)
    return numbers


def summarize_numeric_list(value: list[Any]) -> dict[str, Any]:
    numbers = finite_numbers(value)
    if not numbers:
        return {
            "mean": _nan(),
            "std": _nan(),
            "min": _nan(),
            "max": _nan(),
            "median": _nan(),
            "count": 0,
        }
    return {
        "mean": statistics.fmean(numbers),
        "std": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
        "min": min(numbers),
        "max": max(numbers),
        "median": statistics.median(numbers),
        "count": len(numbers),
    }


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str | int | float):
        return value
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def derive_path_metadata(root: Path, summary_path: Path) -> dict[str, Any]:
    rel = summary_path.relative_to(root)
    parts = rel.parts
    if len(parts) < 3 or parts[-1] != "summary.json":
        raise ValueError(f"Unexpected summary path: {summary_path}")
    if not SEED_RE.match(parts[-2]):
        raise ValueError(f"Summary is not in a seed_<seed> directory: {summary_path}")

    path_method = parts[0]
    experiment = parts[-3] if len(parts) >= 3 else ""
    if path_method == "npe_full_data":
        path_problem = experiment.removesuffix("_full_data")
        path_missingness = "none"
        path_epsilon = 0.0
    else:
        path_problem = parts[1] if len(parts) > 1 else ""
        path_missingness = parts[2] if len(parts) > 2 else ""
        path_epsilon = parse_epsilon_from_text(experiment)

    return {
        "budget": budget_from_root(root),
        "method": path_method,
        "problem": path_problem,
        "missingness": path_missingness,
        "epsilon": path_epsilon,
        "seed": seed_from_path(summary_path),
        "experiment": experiment,
        "run_dir": str(summary_path.parent),
        "summary_path": str(summary_path),
    }


def normalized_identifier(
    key: str,
    summary: dict[str, Any],
    path_metadata: dict[str, Any],
) -> Any:
    value = summary.get(key)
    if key == "method" and not isinstance(value, str):
        return path_metadata[key]
    if key == "problem" and not isinstance(value, str):
        return path_metadata[key]
    if key == "missingness":
        if isinstance(value, str) and value:
            return value
        return "none" if path_metadata["method"] == "npe_full_data" else path_metadata[key]
    if key == "epsilon":
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            pass
        return path_metadata[key]
    if key == "seed":
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
        return path_metadata[key]
    return path_metadata[key]


def build_row(root: Path, summary_path: Path) -> dict[str, Any]:
    path_metadata = derive_path_metadata(root, summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"Summary JSON must be an object: {summary_path}")

    row: dict[str, Any] = {
        **path_metadata,
        "compile_status": "ok",
        "compile_error": "",
    }
    for key in ("method", "problem", "missingness", "epsilon", "seed"):
        row[key] = normalized_identifier(key, summary, path_metadata)

    for key, value in sorted(summary.items()):
        if key in {"method", "problem", "missingness", "epsilon", "seed"}:
            continue
        if is_scalar(value):
            row[key] = csv_value(value)
        else:
            row[key] = csv_value(value)
            if is_numeric_sequence(value):
                stats = summarize_numeric_list(value)
                for stat_name in LIST_SUMMARY_STATS:
                    row[f"{key}_{stat_name}"] = stats[stat_name]
    return row


def build_error_row(root: Path, summary_path: Path, exc: Exception) -> dict[str, Any]:
    try:
        path_metadata = derive_path_metadata(root, summary_path)
    except Exception:
        path_metadata = {
            "budget": budget_from_root(root),
            "method": "",
            "problem": "",
            "missingness": "",
            "epsilon": _nan(),
            "seed": seed_from_path(summary_path),
            "experiment": "",
            "run_dir": str(summary_path.parent),
            "summary_path": str(summary_path),
        }
    return {
        **path_metadata,
        "compile_status": "failed",
        "compile_error": str(exc),
    }


def discover_summary_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("summary.json")
        if path.parent.name.startswith("seed_")
    )


def compile_rows(roots: list[Path], *, strict: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        for summary_path in discover_summary_paths(root):
            try:
                rows.append(build_row(root, summary_path))
            except Exception as exc:
                if strict:
                    raise
                rows.append(build_error_row(root, summary_path, exc))
    return rows


def ordered_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    keys = set().union(*(row.keys() for row in rows)) if rows else set()
    remaining = sorted(keys.difference(IDENTIFIER_COLUMNS))
    return [column for column in IDENTIFIER_COLUMNS if column in keys] + remaining


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ordered_fieldnames(rows)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]], roots: list[Path], out_path: Path | None) -> None:
    ok_rows = sum(row.get("compile_status") == "ok" for row in rows)
    failed_rows = len(rows) - ok_rows
    configs = {
        (
            row.get("budget"),
            row.get("method"),
            row.get("problem"),
            row.get("missingness"),
            row.get("epsilon"),
        )
        for row in rows
        if row.get("compile_status") == "ok"
    }
    print("Experiment result compilation summary")
    print(f"Output roots: {', '.join(str(root) for root in roots)}")
    print(f"Rows: {len(rows)}")
    print(f"Successful rows: {ok_rows}")
    print(f"Failed rows: {failed_rows}")
    print(f"Unique budget/method/problem/missingness/epsilon configs: {len(configs)}")
    if out_path is not None:
        print(f"CSV output: {out_path}")


def main() -> None:
    args = parse_args()
    roots = args.outputs_dirs if args.outputs_dirs else list(DEFAULT_OUTPUT_ROOTS)
    rows = compile_rows(roots, strict=args.strict)
    if args.dry_run:
        print_summary(rows, roots, None)
        return
    write_csv(rows, args.out)
    print_summary(rows, roots, args.out)


if __name__ == "__main__":
    main()
