from __future__ import annotations

import argparse
import csv
import json
from math import ceil
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np

from gapsbi.io import ARRAYS, SPLITS


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal diagnostics for one GAPSBI dataset split.")
    parser.add_argument("--dataset-path", required=True, help="Path to a GAPSBI HDF5 dataset.")
    parser.add_argument("--output-dir", required=True, help="Directory where diagnostic outputs will be written.")
    parser.add_argument("--split", choices=SPLITS, default="train")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)

    split_data, metadata_attrs = load_split(dataset_path, args.split)
    validate_split(split_data, args.split)
    output_dir.mkdir(parents=True, exist_ok=True)

    theta = split_data["theta"]
    x_full = split_data["x_full"]
    mask = split_data["mask"].astype(bool)
    mask_flat = mask.reshape(mask.shape[0], -1)

    save_dataset_summary(
        output_dir / "dataset_summary.json",
        dataset_path=dataset_path,
        split=args.split,
        theta=theta,
        x_full=x_full,
        mask_flat=mask_flat,
        metadata_attrs=metadata_attrs,
    )
    save_theta_outputs(theta, output_dir / "prior")
    save_missingness_outputs(mask_flat, output_dir / "missingness")

    print(f"Dataset: {dataset_path}")
    print(f"Split: {args.split}")
    print(f"Examples: {theta.shape[0]}")
    print(f"Empirical missing fraction: {1.0 - float(mask_flat.mean()):.6f}")
    print(f"Output directory: {output_dir}")


def load_split(dataset_path: Path, split: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load one split and available HDF5 attributes from a GAPSBI dataset."""
    with h5py.File(dataset_path, "r") as h5:
        if split not in h5:
            raise ValueError(f"Split {split!r} not found in {dataset_path}.")

        missing_arrays = [name for name in ARRAYS if name not in h5[split]]
        if missing_arrays:
            raise ValueError(f"Split {split!r} is missing arrays: {missing_arrays}.")

        split_data = {name: h5[split][name][...] for name in ARRAYS}
        metadata_attrs = {
            "file": attrs_to_jsonable_dict(h5.attrs),
            "split": attrs_to_jsonable_dict(h5[split].attrs),
        }

    return split_data, metadata_attrs


def validate_split(split_data: dict[str, np.ndarray], split: str) -> None:
    """Validate the minimal shape and mask contract needed by these diagnostics."""
    theta = np.asarray(split_data["theta"])
    x_full = np.asarray(split_data["x_full"])
    x_obs = np.asarray(split_data["x_obs"])
    mask = np.asarray(split_data["mask"])

    if theta.ndim != 2:
        raise ValueError(f"{split}/theta must be 2D, got shape {theta.shape}.")
    if x_full.ndim < 2:
        raise ValueError(f"{split}/x_full must include an example axis and feature axes, got shape {x_full.shape}.")
    if x_full.shape != x_obs.shape or x_full.shape != mask.shape:
        raise ValueError(
            f"{split}/x_full, x_obs, and mask must have matching shapes; "
            f"got {x_full.shape}, {x_obs.shape}, and {mask.shape}."
        )
    if theta.shape[0] != x_full.shape[0]:
        raise ValueError(f"{split}/theta and x arrays must have the same number of examples.")
    if not np.all(np.isfinite(theta)):
        raise ValueError(f"{split}/theta must contain only finite values.")
    if not np.all(np.isfinite(x_full)) or not np.all(np.isfinite(x_obs)):
        raise ValueError(f"{split}/x_full and x_obs must contain only finite values.")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError(f"{split}/mask must be binary.")
    if not np.array_equal(x_obs, x_full * mask):
        raise ValueError(f"{split}/x_obs must equal x_full * mask.")


def save_dataset_summary(
    output_path: Path,
    *,
    dataset_path: Path,
    split: str,
    theta: np.ndarray,
    x_full: np.ndarray,
    mask_flat: np.ndarray,
    metadata_attrs: dict[str, Any],
) -> None:
    summary = {
        "dataset_path": str(dataset_path),
        "split": split,
        "number_of_examples": int(theta.shape[0]),
        "theta_dimension": int(theta.shape[1]),
        "x_dimension": int(np.prod(x_full.shape[1:])),
        "x_shape": list(x_full.shape[1:]),
        "empirical_missing_fraction": 1.0 - float(mask_flat.mean()),
        "hdf5_attrs": metadata_attrs,
    }
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_theta_outputs(theta: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(output_dir / "theta_summary.csv", per_feature_summary(theta, prefix="theta"))
    plot_theta_marginals(theta, output_dir / "theta_marginals.png")


def save_missingness_outputs(mask_flat: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_by_feature = 1.0 - mask_flat.mean(axis=0)
    missing_by_example = 1.0 - mask_flat.mean(axis=1)

    rows = [
        {"metric": "empirical_missing_fraction", "value": float(missing_by_example.mean())},
        *summary_stat_rows(missing_by_feature, "missing_fraction_by_feature"),
        *summary_stat_rows(missing_by_example, "missing_fraction_by_example"),
    ]
    write_rows_csv(output_dir / "missingness_summary.csv", rows)
    plot_missing_fraction_by_feature(missing_by_feature, output_dir / "missing_fraction_by_feature.png")
    plot_missing_fraction_by_example(missing_by_example, output_dir / "missing_fraction_by_example.png")


def per_feature_summary(values: np.ndarray, prefix: str) -> list[dict[str, float | str]]:
    rows = []
    for index in range(values.shape[1]):
        stats = summary_stats(values[:, index])
        rows.append({"parameter": f"{prefix}_{index}", **stats})
    return rows


def summary_stat_rows(values: np.ndarray, prefix: str) -> list[dict[str, float | str]]:
    return [{"metric": f"{prefix}_{key}", "value": value} for key, value in summary_stats(values).items()]


def summary_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "count": float(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
    }


def write_summary_csv(output_path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty summary CSV.")
    fieldnames = list(rows[0].keys())
    write_rows_csv(output_path, rows, fieldnames=fieldnames)


def write_rows_csv(
    output_path: Path,
    rows: list[dict[str, float | str]],
    fieldnames: list[str] | None = None,
) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV.")
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_theta_marginals(theta: np.ndarray, output_path: Path) -> None:
    theta_dim = theta.shape[1]
    n_cols = min(4, theta_dim)
    n_rows = ceil(theta_dim / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.0 * n_rows), squeeze=False)

    for index, ax in enumerate(axes.ravel()):
        if index >= theta_dim:
            ax.set_visible(False)
            continue
        ax.hist(theta[:, index], bins=40, color="C0", alpha=0.85)
        ax.set_title(f"theta_{index}")
        ax.set_xlabel("value")
        ax.set_ylabel("count")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_missing_fraction_by_feature(missing_by_feature: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 4.0))
    ax.plot(np.arange(missing_by_feature.size), missing_by_feature, linewidth=1.5)
    ax.set_xlabel("feature")
    ax.set_ylabel("missing fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Missing fraction by feature")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_missing_fraction_by_example(missing_by_example: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.hist(missing_by_example, bins=40, color="C1", alpha=0.85)
    ax.set_xlabel("missing fraction")
    ax.set_ylabel("number of examples")
    ax.set_xlim(-0.02, 1.02)
    ax.set_title("Missing fraction by example")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def attrs_to_jsonable_dict(attrs: h5py.AttributeManager) -> dict[str, Any]:
    return {key: attr_to_jsonable(value) for key, value in attrs.items()}


def attr_to_jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return attr_to_jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [attr_to_jsonable(item) for item in value.tolist()]
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(value, list | tuple):
        return [attr_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): attr_to_jsonable(item) for key, item in value.items()}
    return value


if __name__ == "__main__":
    main()
