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
    x_full_flat = x_full.reshape(x_full.shape[0], -1)
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
    save_missingness_outputs(mask_flat, x_full_flat, output_dir / "missingness")
    save_observation_outputs(x_full_flat, output_dir / "observations")
    save_theta_x_outputs(theta, x_full_flat, output_dir / "theta_x")

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


def save_missingness_outputs(mask_flat: np.ndarray, x_full_flat: np.ndarray, output_dir: Path) -> None:
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
    plot_missingness_vs_x_value(x_full_flat.ravel(), mask_flat.ravel(), output_dir / "missingness_vs_x_value.png")


def save_observation_outputs(x_full_flat: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_mean = x_full_flat.mean(axis=0)
    feature_std = x_full_flat.std(axis=0)
    feature_min = x_full_flat.min(axis=0)
    feature_max = x_full_flat.max(axis=0)
    quantiles = np.percentile(x_full_flat, [5, 25, 50, 75, 95], axis=0)
    near_constant_count = int(np.sum(feature_std < 1e-8))

    write_observation_summary_csv(
        output_dir / "observation_summary.csv",
        x_full_flat=x_full_flat,
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_min=feature_min,
        feature_max=feature_max,
        near_constant_count=near_constant_count,
    )
    plot_x_feature_mean_std(feature_mean, feature_std, output_dir / "x_feature_mean_std.png")
    plot_x_feature_quantiles(quantiles, output_dir / "x_feature_quantiles.png")
    plot_x_global_distribution(x_full_flat.ravel(), output_dir / "x_global_distribution.png")


def save_theta_x_outputs(theta: np.ndarray, x_full_flat: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    correlations, constant_feature_count = compute_theta_x_correlations(theta, x_full_flat)

    if constant_feature_count:
        print(f"Warning: set correlations to zero for {constant_feature_count} near-constant x_full features.")

    write_theta_x_summary_csv(output_dir / "theta_x_summary.csv", correlations)
    plot_theta_x_correlation_heatmap(correlations, output_dir / "theta_x_correlation_heatmap.png")


def compute_theta_x_correlations(
    theta: np.ndarray,
    x_full_flat: np.ndarray,
    constant_threshold: float = 1e-8,
) -> tuple[np.ndarray, int]:
    theta_centered = theta - theta.mean(axis=0, keepdims=True)
    x_centered = x_full_flat - x_full_flat.mean(axis=0, keepdims=True)

    theta_scale = np.sqrt(np.sum(theta_centered**2, axis=0))
    x_scale = np.sqrt(np.sum(x_centered**2, axis=0))
    valid_theta = theta_scale >= constant_threshold
    valid_x = x_scale >= constant_threshold

    correlations = np.zeros((theta.shape[1], x_full_flat.shape[1]), dtype=float)
    if np.any(valid_theta) and np.any(valid_x):
        numerator = theta_centered[:, valid_theta].T @ x_centered[:, valid_x]
        denominator = theta_scale[valid_theta][:, None] * x_scale[valid_x][None, :]
        correlations[np.ix_(valid_theta, valid_x)] = numerator / denominator

    return correlations, int(np.sum(~valid_x))


def write_theta_x_summary_csv(output_path: Path, correlations: np.ndarray, top_k: int = 5) -> None:
    rows: list[dict[str, float | int | str]] = []
    abs_correlations = np.abs(correlations)

    for theta_index in range(correlations.shape[0]):
        abs_row = abs_correlations[theta_index]
        top_count = min(top_k, correlations.shape[1])
        top_indices = np.argsort(abs_row)[::-1][:top_count]
        rows.append(
            {
                "theta_dim": theta_index,
                "max_abs_correlation": float(abs_row[top_indices[0]]) if top_count else 0.0,
                "max_abs_correlation_feature": int(top_indices[0]) if top_count else "",
                "mean_abs_correlation": float(abs_row.mean()) if abs_row.size else 0.0,
                "top_feature_indices": ";".join(str(int(index)) for index in top_indices),
                "top_correlations": ";".join(f"{float(correlations[theta_index, index]):.6g}" for index in top_indices),
            }
        )

    write_rows_csv(
        output_path,
        rows,
        fieldnames=[
            "theta_dim",
            "max_abs_correlation",
            "max_abs_correlation_feature",
            "mean_abs_correlation",
            "top_feature_indices",
            "top_correlations",
        ],
    )


def write_observation_summary_csv(
    output_path: Path,
    *,
    x_full_flat: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    feature_min: np.ndarray,
    feature_max: np.ndarray,
    near_constant_count: int,
) -> None:
    rows: list[dict[str, float | int | str]] = [
        {
            "scope": "global",
            "feature": "",
            "mean": float(x_full_flat.mean()),
            "std": float(x_full_flat.std()),
            "min": float(x_full_flat.min()),
            "max": float(x_full_flat.max()),
            "near_constant_features": near_constant_count,
        }
    ]

    for index in range(x_full_flat.shape[1]):
        rows.append(
            {
                "scope": "feature",
                "feature": index,
                "mean": float(feature_mean[index]),
                "std": float(feature_std[index]),
                "min": float(feature_min[index]),
                "max": float(feature_max[index]),
                "near_constant_features": "",
            }
        )

    write_rows_csv(
        output_path,
        rows,
        fieldnames=["scope", "feature", "mean", "std", "min", "max", "near_constant_features"],
    )


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


def plot_missingness_vs_x_value(
    x_values: np.ndarray,
    mask_values: np.ndarray,
    output_path: Path,
    n_bins: int = 30,
) -> None:
    x_values = np.asarray(x_values, dtype=float)
    missing = 1.0 - np.asarray(mask_values, dtype=float)
    unique_values = np.unique(x_values)

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    if unique_values.size < 2:
        ax.text(
            0.5,
            0.5,
            "Skipped: fewer than 2 unique x_full values",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        print("Warning: skipping missingness-vs-x binning because x_full has fewer than 2 unique values.")
    else:
        bins = np.linspace(float(x_values.min()), float(x_values.max()), min(n_bins, unique_values.size) + 1)
        bin_index = np.digitize(x_values, bins[1:-1], right=False)
        bin_count = np.bincount(bin_index, minlength=bins.size - 1)
        missing_count = np.bincount(bin_index, weights=missing, minlength=bins.size - 1)
        valid = bin_count > 0
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        missing_fraction = np.full(bins.size - 1, np.nan)
        missing_fraction[valid] = missing_count[valid] / bin_count[valid]

        ax.plot(bin_centers[valid], missing_fraction[valid], marker="o", linewidth=1.5)
        ax.axhline(float(missing.mean()), color="0.4", linestyle="--", linewidth=1.0, label="overall")
        ax.legend(fontsize="small")

    ax.set_xlabel("binned x_full value")
    ax.set_ylabel("empirical missing fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Missingness vs x_full value")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_x_feature_mean_std(feature_mean: np.ndarray, feature_std: np.ndarray, output_path: Path) -> None:
    feature = np.arange(feature_mean.size)
    fig, ax = plt.subplots(figsize=(10.0, 4.0))
    ax.plot(feature, feature_mean, color="C0", linewidth=1.5, label="mean")
    ax.fill_between(
        feature,
        feature_mean - feature_std,
        feature_mean + feature_std,
        color="C0",
        alpha=0.2,
        label="mean +/- std",
    )
    ax.set_xlabel("feature")
    ax.set_ylabel("x_full")
    ax.set_title("x_full feature mean +/- std")
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_x_feature_quantiles(quantiles: np.ndarray, output_path: Path) -> None:
    feature = np.arange(quantiles.shape[1])
    q05, q25, q50, q75, q95 = quantiles

    fig, ax = plt.subplots(figsize=(10.0, 4.0))
    ax.fill_between(feature, q05, q95, color="C0", alpha=0.15, label="5%-95%")
    ax.fill_between(feature, q25, q75, color="C0", alpha=0.3, label="25%-75%")
    ax.plot(feature, q50, color="C0", linewidth=1.5, label="median")
    ax.set_xlabel("feature")
    ax.set_ylabel("x_full")
    ax.set_title("x_full feature quantile bands")
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_x_global_distribution(values: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.hist(values, bins=80, color="C2", alpha=0.85)
    ax.set_xlabel("x_full value")
    ax.set_ylabel("count")
    ax.set_title("Global x_full distribution")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_theta_x_correlation_heatmap(correlations: np.ndarray, output_path: Path) -> None:
    n_theta, n_features = correlations.shape
    fig_width = max(8.0, min(18.0, 0.25 * n_features))
    fig_height = max(3.0, min(10.0, 0.6 * n_theta + 1.5))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    image = ax.imshow(correlations, aspect="auto", vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xlabel("feature")
    ax.set_ylabel("theta dimension")
    ax.set_title("Pearson correlation: theta vs x_full features")
    ax.set_yticks(np.arange(n_theta))
    ax.set_yticklabels([str(index) for index in range(n_theta)])
    fig.colorbar(image, ax=ax, label="correlation")
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
