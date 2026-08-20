from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


DEFAULT_INPUT_PATH = Path("outputs/analysis/full_experiment_results.csv")
DEFAULT_OUTPUT_PATH = Path("outputs/analysis/full_experiment_results_with_diagnostics.csv")
DIAGNOSTIC_COLUMNS = [
    "tarp_mae",
    "tarp_iae",
    "sbc_ks_pval_min",
    "sbc_ks_pval_mean",
    "diagnostics_status",
    "diagnostics_error",
    "diagnostics_arrays_resolved_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add diagnostics-array-derived metrics to compiled experiment results.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input compiled results CSV. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output enriched results CSV. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise on missing or invalid diagnostics instead of recording failed rows.",
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


def resolve_diagnostics_path(row: dict[str, str]) -> Path:
    candidate = str(row.get("diagnostics_arrays_path", "")).strip()
    if candidate:
        path = Path(candidate)
    else:
        run_dir = str(row.get("run_dir", "")).strip()
        if not run_dir:
            raise ValueError("Missing diagnostics_arrays_path and run_dir.")
        path = Path(run_dir) / "diagnostics_arrays.npz"
    return path if path.is_absolute() else Path.cwd() / path


def compute_tarp_metrics(arrays: Any) -> dict[str, float]:
    if "alpha" not in arrays or "ecp" not in arrays:
        raise ValueError("Diagnostics file is missing alpha/ecp arrays.")
    alpha = np.asarray(arrays["alpha"], dtype=float).reshape(-1)
    ecp = np.asarray(arrays["ecp"], dtype=float).reshape(-1)
    if alpha.shape != ecp.shape:
        raise ValueError(f"alpha/ecp shape mismatch: {alpha.shape} != {ecp.shape}.")
    if alpha.size == 0:
        raise ValueError("alpha/ecp arrays are empty.")
    diff = np.abs(ecp - alpha)
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return {
        "tarp_mae": float(np.mean(diff)),
        "tarp_iae": float(trapezoid(diff, alpha)),
    }


def compute_sbc_metrics(arrays: Any, row: dict[str, str]) -> dict[str, float]:
    if "ranks" not in arrays:
        raise ValueError("Diagnostics file is missing ranks array.")
    num_posterior_samples = parse_float(row.get("num_sbc_posterior_samples"))
    if not np.isfinite(num_posterior_samples) or num_posterior_samples <= 0:
        raise ValueError("Missing or invalid num_sbc_posterior_samples.")

    ranks = np.asarray(arrays["ranks"], dtype=float)
    if ranks.ndim == 1:
        ranks = ranks.reshape(-1, 1)
    if ranks.ndim != 2 or ranks.shape[0] == 0 or ranks.shape[1] == 0:
        raise ValueError(f"Invalid ranks shape: {ranks.shape}.")

    uniformized = (ranks + 0.5) / (num_posterior_samples + 1.0)
    p_values = []
    for dim in range(uniformized.shape[1]):
        values = uniformized[:, dim]
        values = values[np.isfinite(values)]
        if values.size:
            p_values.append(float(stats.kstest(values, "uniform").pvalue))
    if not p_values:
        raise ValueError("No finite SBC rank values.")

    return {
        "sbc_ks_pval_min": float(min(p_values)),
        "sbc_ks_pval_mean": float(np.mean(p_values)),
    }


def compute_diagnostic_metrics(row: dict[str, str]) -> dict[str, Any]:
    path = resolve_diagnostics_path(row)
    if not path.exists():
        raise FileNotFoundError(f"Diagnostics file not found: {path}")

    with np.load(path) as arrays:
        return {
            **compute_tarp_metrics(arrays),
            **compute_sbc_metrics(arrays, row),
            "diagnostics_status": "ok",
            "diagnostics_error": "",
            "diagnostics_arrays_resolved_path": str(path),
        }


def empty_diagnostic_metrics(status: str, error: str, row: dict[str, str]) -> dict[str, Any]:
    try:
        path = resolve_diagnostics_path(row)
        resolved_path = str(path)
    except Exception:
        resolved_path = ""
    return {
        "tarp_mae": "",
        "tarp_iae": "",
        "sbc_ks_pval_min": "",
        "sbc_ks_pval_mean": "",
        "diagnostics_status": status,
        "diagnostics_error": error,
        "diagnostics_arrays_resolved_path": resolved_path,
    }


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def enrich_rows(rows: list[dict[str, str]], *, strict: bool = False) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        output_row: dict[str, Any] = dict(row)
        try:
            output_row.update(compute_diagnostic_metrics(row))
        except Exception as exc:
            if strict:
                raise
            status = "missing_file" if isinstance(exc, FileNotFoundError) else "failed"
            output_row.update(empty_diagnostic_metrics(status, str(exc), row))
        enriched.append(output_row)
    return enriched


def output_fieldnames(input_fieldnames: list[str]) -> list[str]:
    fields = [field for field in input_fieldnames if field not in DIAGNOSTIC_COLUMNS]
    fields.extend(DIAGNOSTIC_COLUMNS)
    return fields


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]], input_path: Path, output_path: Path) -> None:
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("diagnostics_status", ""))
        status_counts[status] = status_counts.get(status, 0) + 1
    print("Diagnostic metrics enrichment summary")
    print(f"Input: {input_path}")
    print(f"Rows: {len(rows)}")
    print(f"Status counts: {status_counts}")
    print(f"Output: {output_path}")


def main() -> None:
    args = parse_args()
    rows, fieldnames = read_csv(args.input)
    enriched = enrich_rows(rows, strict=args.strict)
    write_csv(args.out, enriched, output_fieldnames(fieldnames))
    print_summary(enriched, args.input, args.out)


if __name__ == "__main__":
    main()
