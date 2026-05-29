from __future__ import annotations

import argparse
import csv
import ast
import json
import math
import re
import struct
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


METHOD_NAMES = {
    "npe_full_data": "full_data",
    "npe_zero_imputation": "zero_imputation",
    "npe_mean_imputation": "mean_imputation",
    "npe_mask_augmentation": "mean_imputation_mask",
}

REQUIRED_COLUMNS = [
    "method",
    "problem",
    "missingness",
    "epsilon",
    "experiment",
    "seed",
    "run_dir",
    "status",
    "tarp_mae",
    "tarp_iae",
    "sbc_ks_pval_min",
    "sbc_ks_pval_mean",
    "training_time_sec",
    "posterior_sampling_time_sec",
    "diagnostics_time_sec",
    "total_runtime_sec",
]

EPSILON_RE = re.compile(r"eps(\d{3})")
SEED_RE = re.compile(r"^seed_(\d+)$")


def _nan() -> float:
    return float("nan")


def parse_epsilon(text: str) -> float:
    match = EPSILON_RE.search(text)
    if match is None:
        return _nan()
    return int(match.group(1)) / 100.0


def parse_result_path(summary_path: Path, outputs_dir: Path) -> dict[str, Any]:
    """Parse Campaign 1 metadata from a seed-level summary path."""
    rel = summary_path.relative_to(outputs_dir)
    parts = rel.parts
    if len(parts) < 4 or parts[-1] != "summary.json":
        raise ValueError(f"Not a seed summary path: {summary_path}")

    seed_dir = parts[-2]
    seed_match = SEED_RE.match(seed_dir)
    if seed_match is None:
        raise ValueError(f"Summary is not inside a seed_<seed> directory: {summary_path}")

    raw_method = parts[0]
    method = METHOD_NAMES.get(raw_method, raw_method)
    seed = int(seed_match.group(1))
    run_dir = summary_path.parent

    if raw_method == "npe_full_data":
        experiment = parts[1]
        problem = experiment.removesuffix("_full_data")
        missingness = "none"
        epsilon = 0.0
    else:
        if len(parts) < 6:
            raise ValueError(f"Missing-data summary path is too shallow: {summary_path}")
        problem = parts[1]
        missingness = parts[2]
        experiment = parts[3]
        epsilon = parse_epsilon(experiment)

    return {
        "method": method,
        "problem": problem,
        "missingness": missingness,
        "epsilon": epsilon,
        "experiment": experiment,
        "seed": seed,
        "run_dir": str(run_dir),
    }


def _as_float(value: Any) -> float:
    if value is None:
        return _nan()
    try:
        return float(value)
    except (TypeError, ValueError):
        return _nan()


def _ks_uniform_pvalue(values: Any) -> float:
    """Approximate one-sample KS p-value against U(0, 1)."""
    sorted_values = sorted(float(v) for v in values if math.isfinite(float(v)))
    n = len(sorted_values)
    if n == 0:
        return _nan()

    d_plus = max((idx + 1) / n - value for idx, value in enumerate(sorted_values))
    d_minus = max(value - idx / n for idx, value in enumerate(sorted_values))
    d_stat = max(d_plus, d_minus)

    z = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d_stat
    p_value = 0.0
    for k in range(1, 101):
        term = 2.0 * ((-1.0) ** (k - 1)) * math.exp(-2.0 * (k * z) ** 2)
        p_value += term
        if abs(term) < 1e-12:
            break
    return min(1.0, max(0.0, p_value))


def _product(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _parse_npy_bytes(payload: bytes) -> tuple[list[float], tuple[int, ...]]:
    """Read a simple numeric .npy payload without importing NumPy."""
    if not payload.startswith(b"\x93NUMPY"):
        raise ValueError("Invalid .npy payload")

    major = payload[6]
    if major == 1:
        header_len = struct.unpack("<H", payload[8:10])[0]
        data_offset = 10 + header_len
    elif major in {2, 3}:
        header_len = struct.unpack("<I", payload[8:12])[0]
        data_offset = 12 + header_len
    else:
        raise ValueError(f"Unsupported .npy version: {major}")

    header = ast.literal_eval(payload[data_offset - header_len : data_offset].decode("latin1"))
    if header.get("fortran_order"):
        raise ValueError("Fortran-order .npy arrays are not supported")

    descr = str(header["descr"])
    shape = tuple(int(dim) for dim in header["shape"])
    count = _product(shape)
    if count == 0:
        return [], shape

    endian = ">" if descr.startswith(">") else "<"
    kind = descr[-2]
    size = int(descr[-1])
    format_map = {
        ("f", 4): "f",
        ("f", 8): "d",
        ("i", 1): "b",
        ("i", 2): "h",
        ("i", 4): "i",
        ("i", 8): "q",
        ("u", 1): "B",
        ("u", 2): "H",
        ("u", 4): "I",
        ("u", 8): "Q",
    }
    fmt = format_map.get((kind, size))
    if fmt is None:
        raise ValueError(f"Unsupported .npy dtype: {descr}")

    data = payload[data_offset : data_offset + count * size]
    return list(struct.unpack(f"{endian}{count}{fmt}", data)), shape


def _load_npz_arrays(path: Path, names: set[str]) -> dict[str, tuple[list[float], tuple[int, ...]]]:
    arrays: dict[str, tuple[list[float], tuple[int, ...]]] = {}
    with zipfile.ZipFile(path) as archive:
        for name in names:
            npy_name = f"{name}.npy"
            if npy_name in archive.namelist():
                arrays[name] = _parse_npy_bytes(archive.read(npy_name))
    return arrays


def compute_diagnostics_metrics(
    summary: dict[str, Any],
    seed_dir: Path,
    outputs_dir: Path,
) -> dict[str, float]:
    """Compute diagnostics-derived metrics, returning NaNs if unavailable."""
    metrics = {
        "tarp_mae": _nan(),
        "tarp_iae": _nan(),
        "sbc_ks_pval_min": _nan(),
        "sbc_ks_pval_mean": _nan(),
    }

    diagnostics_path = summary.get("diagnostics_arrays_path")
    if diagnostics_path:
        diagnostics_file = Path(str(diagnostics_path))
        if not diagnostics_file.is_absolute():
            diagnostics_file = outputs_dir.parent / diagnostics_file
    else:
        diagnostics_file = seed_dir / "diagnostics_arrays.npz"

    if not diagnostics_file.exists():
        return metrics

    try:
        arrays = _load_npz_arrays(diagnostics_file, {"alpha", "ecp", "ranks"})
    except Exception:
        return metrics

    if {"alpha", "ecp"}.issubset(arrays):
        alpha, alpha_shape = arrays["alpha"]
        ecp, ecp_shape = arrays["ecp"]
        if alpha_shape == ecp_shape and alpha:
            diff = [abs(ecp_value - alpha_value) for ecp_value, alpha_value in zip(ecp, alpha)]
            metrics["tarp_mae"] = float(math.fsum(diff) / len(diff))
            metrics["tarp_iae"] = float(
                math.fsum(
                    0.5 * (diff[idx] + diff[idx - 1]) * (alpha[idx] - alpha[idx - 1])
                    for idx in range(1, len(diff))
                )
            )

    if "ranks" in arrays:
        ranks, ranks_shape = arrays["ranks"]
        num_posterior_samples = _as_float(summary.get("num_sbc_posterior_samples"))
        if len(ranks_shape) == 1:
            num_rows, num_dims = ranks_shape[0], 1
        elif len(ranks_shape) == 2:
            num_rows, num_dims = ranks_shape
        else:
            num_rows, num_dims = 0, 0

        if num_rows > 0 and num_dims > 0 and math.isfinite(num_posterior_samples):
            denom = num_posterior_samples + 1.0
            p_values = [
                _ks_uniform_pvalue(
                    (ranks[row_idx * num_dims + dim] + 0.5) / denom
                    for row_idx in range(num_rows)
                )
                for dim in range(num_dims)
            ]
            p_values = [p for p in p_values if math.isfinite(p)]
            if p_values:
                metrics["sbc_ks_pval_min"] = float(min(p_values))
                metrics["sbc_ks_pval_mean"] = float(sum(p_values) / len(p_values))

    return metrics


def discover_summary_paths(outputs_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in outputs_dir.rglob("summary.json")
        if path.parent.name.startswith("seed_") and path.name != ".DS_Store"
        and path.relative_to(outputs_dir).parts[0] in METHOD_NAMES
    )


def build_row(summary_path: Path, outputs_dir: Path) -> dict[str, Any]:
    parsed = parse_result_path(summary_path, outputs_dir)
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        summary = {"status": f"failed_to_read_summary: {exc}"}

    seed_dir = summary_path.parent
    diagnostics = compute_diagnostics_metrics(summary, seed_dir, outputs_dir)
    status = str(summary.get("status", "ok"))

    row = {
        **parsed,
        "status": status,
        **diagnostics,
        "training_time_sec": _as_float(summary.get("training_time_sec")),
        "posterior_sampling_time_sec": _as_float(summary.get("posterior_sampling_time_sec")),
        "diagnostics_time_sec": _as_float(summary.get("diagnostics_time_sec")),
        "total_runtime_sec": _as_float(summary.get("total_runtime_sec")),
    }

    # Prefer explicit JSON values if present, but keep path parsing as the source
    # of truth for full-data runs and legacy summaries without these fields.
    row["seed"] = int(summary.get("seed", row["seed"]))
    row["problem"] = str(summary.get("problem", row["problem"]))
    return {column: row.get(column, _nan()) for column in REQUIRED_COLUMNS}


def build_results_table(outputs_dir: Path) -> list[dict[str, Any]]:
    return [build_row(path, outputs_dir) for path in discover_summary_paths(outputs_dir)]


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_parquet_if_available(rows: list[dict[str, Any]], csv_path: Path) -> Path | None:
    parquet_path = csv_path.with_suffix(".parquet")
    code = """
import json
import sys
import pandas as pd

rows = json.loads(sys.stdin.read())
pd.DataFrame(rows).to_parquet(sys.argv[1], index=False)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(parquet_path)],
        input=json.dumps(rows),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        reason = result.stderr.strip() or f"parquet subprocess exited {result.returncode}"
        print(f"Skipping parquet output: {reason}")
        return None
    return parquet_path


def is_ok_status(status: Any) -> bool:
    return str(status).strip().lower() in {"ok", "success", "completed"}


def print_summary(rows: list[dict[str, Any]], out_path: Path | None, parquet_path: Path | None) -> None:
    unique_configs = {
        (
            row["method"],
            row["problem"],
            row["missingness"],
            row["epsilon"],
            row["experiment"],
        )
        for row in rows
    }
    unique_seeds = {row["seed"] for row in rows}
    failed_runs = sum(1 for row in rows if not is_ok_status(row["status"]))

    print("Campaign 1 aggregation summary")
    print(f"Rows: {len(rows)}")
    print(f"Unique configs: {len(unique_configs)}")
    print(f"Unique seeds: {len(unique_seeds)}")
    print(f"Failed/non-ok runs: {failed_runs}")
    if out_path is not None:
        print(f"CSV output: {out_path}")
    if parquet_path is not None:
        print(f"Parquet output: {parquet_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Campaign 1 seed-level NPE results from outputs/.",
    )
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/campaign1_master_results.csv"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the table and print the summary without writing files.",
    )
    parser.add_argument(
        "--no-parquet",
        action="store_true",
        help="Only write CSV, even if parquet support is installed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_results_table(args.outputs_dir)

    if args.dry_run:
        print_summary(rows, None, None)
        return

    write_csv(rows, args.out)
    parquet_path = None if args.no_parquet else write_parquet_if_available(rows, args.out)
    print_summary(rows, args.out, parquet_path)


if __name__ == "__main__":
    main()
