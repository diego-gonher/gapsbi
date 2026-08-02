from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample train examples and diagnose how much information masks remove."
    )
    parser.add_argument("datasets", nargs="*", help="HDF5 datasets to diagnose.")
    parser.add_argument(
        "--root",
        default="data/canonical_v1",
        help="Dataset root used when explicit datasets are omitted.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/dataset_generation_diagnostics",
        help="Root directory for diagnostic outputs.",
    )
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset_paths = [Path(path) for path in args.datasets]
    if not dataset_paths:
        dataset_paths = sorted(Path(args.root).glob("*/*/*.h5"))
    if not dataset_paths:
        raise ValueError("No datasets found.")

    output_root = Path(args.output_root)
    rows = []
    for dataset_path in dataset_paths:
        row = diagnose_dataset(
            dataset_path=dataset_path,
            output_root=output_root,
            sample_size=args.sample_size,
            seed=args.seed,
        )
        rows.append(row)

    aggregate_path = output_root / "mask_information_summary.csv"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with aggregate_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Datasets: {len(rows)}")
    print(f"Aggregate summary: {aggregate_path}")


def diagnose_dataset(
    *,
    dataset_path: Path,
    output_root: Path,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    problem = dataset_path.parts[-3]
    stem = dataset_path.stem
    output_dir = output_root / problem / stem / "mask_information"
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(dataset_path, "r") as h5:
        metadata = decode_attrs(h5.attrs)
        split = h5["train"]
        n = split["x_full"].shape[0]
        rng = np.random.default_rng(seed)
        count = min(sample_size, n)
        indices = np.sort(rng.choice(n, size=count, replace=False))
        theta = split["theta"][indices]
        x_full = split["x_full"][indices]
        mask = split["mask"][indices].astype(bool)

    x_flat = x_full.reshape((x_full.shape[0], -1)).astype(float)
    mask_flat = mask.reshape((mask.shape[0], -1))
    missing_flat = ~mask_flat
    score = score_for_dataset(problem, x_full, metadata)
    score_flat = score.reshape(-1)
    missing_score_flat = missing_flat.reshape(-1)

    observed_values = x_flat[mask_flat]
    missing_values = x_flat[missing_flat]
    missing_by_feature = missing_flat.mean(axis=0)
    missing_by_example = missing_flat.mean(axis=1)
    low_rate, high_rate, high_gap = low_high_score_missing_rates(score_flat, missing_score_flat)

    row: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "problem": problem,
        "dataset": stem,
        "sample_size": int(count),
        "target_missing_fraction_parameter": get_nested(metadata, "mask", "missing_fraction"),
        "mask_name": get_nested(metadata, "mask", "name"),
        "realized_missing_fraction_sample": float(missing_flat.mean()),
        "missing_fraction_by_example_p95": float(np.quantile(missing_by_example, 0.95)),
        "all_missing_example_rate": float(np.mean(missing_by_example == 1.0)),
        "nearly_all_missing_example_rate": float(np.mean(missing_by_example >= 0.9)),
        "missing_fraction_by_feature_min": float(np.min(missing_by_feature)),
        "missing_fraction_by_feature_max": float(np.max(missing_by_feature)),
        "observed_value_median": safe_quantile(observed_values, 0.5),
        "missing_value_median": safe_quantile(missing_values, 0.5),
        "low_score_missing_rate": low_rate,
        "high_score_missing_rate": high_rate,
        "high_minus_low_score_missing_rate": high_gap,
    }
    row.update(problem_specific_metrics(problem, x_full, mask, score))

    (output_dir / "summary.json").write_text(
        json.dumps(json_ready(row), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_missing_by_feature(missing_by_feature, output_dir / "missing_fraction_by_feature.png", problem)
    plot_observed_missing_hist(observed_values, missing_values, output_dir / "observed_vs_missing_values.png")
    plot_score_missing_curve(score_flat, missing_score_flat, output_dir / "score_missing_curve.png")

    print(f"{dataset_path}: missing={row['realized_missing_fraction_sample']:.4f}, out={output_dir}")
    return row


def decode_attrs(attrs: h5py.AttributeManager) -> dict[str, Any]:
    decoded = {}
    for key, value in attrs.items():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        if isinstance(value, np.generic):
            value = value.item()
        decoded[key] = value
    return decoded


def get_nested(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return ""
        value = value[key]
    return value


def score_for_dataset(problem: str, x_full: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    if problem == "lotka_volterra":
        reshaped = x_full.reshape((x_full.shape[0], -1, 2)).astype(float)
        totals = np.log(np.sum(reshaped, axis=-1) + 1e-12)
        time_score = mean_normalized_minmax(totals)
        return np.repeat(time_score[:, :, None], reshaped.shape[-1], axis=2).reshape(x_full.shape)
    if problem == "glm":
        return x_full.astype(float)
    return mean_normalized_minmax(x_full.reshape((x_full.shape[0], -1)).astype(float)).reshape(x_full.shape)


def mean_normalized_minmax(values: np.ndarray) -> np.ndarray:
    flat = values.reshape((values.shape[0], -1)).astype(float)
    mins = flat.min(axis=1, keepdims=True)
    maxs = flat.max(axis=1, keepdims=True)
    ranges = maxs - mins
    raw = np.zeros_like(flat, dtype=float)
    valid = ranges[:, 0] > 1e-12
    raw[valid] = (flat[valid] - mins[valid]) / (ranges[valid] + 1e-12)
    means = raw.mean(axis=1, keepdims=True)
    scores = np.ones_like(flat, dtype=float)
    valid_mean = valid & (means[:, 0] > 1e-12)
    scores[valid_mean] = raw[valid_mean] / (means[valid_mean] + 1e-12)
    return scores.reshape(values.shape)


def low_high_score_missing_rates(score_flat: np.ndarray, missing_flat: np.ndarray) -> tuple[float, float, float]:
    if score_flat.size == 0:
        return float("nan"), float("nan"), float("nan")
    low_q, high_q = np.quantile(score_flat, [0.2, 0.8])
    low = score_flat <= low_q
    high = score_flat >= high_q
    low_rate = float(np.mean(missing_flat[low])) if np.any(low) else float("nan")
    high_rate = float(np.mean(missing_flat[high])) if np.any(high) else float("nan")
    return low_rate, high_rate, high_rate - low_rate


def problem_specific_metrics(problem: str, x_full: np.ndarray, mask: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if problem == "glm":
        spikes = x_full.astype(bool)
        observed_spikes = spikes & mask
        metrics["glm_total_spike_rate"] = float(np.mean(spikes))
        metrics["glm_spike_retention_rate"] = safe_ratio(np.sum(observed_spikes), np.sum(spikes))
        metrics["glm_masked_spike_rate"] = safe_ratio(np.sum(spikes & ~mask), np.sum(spikes))
    if problem == "lotka_volterra":
        paired = mask.reshape((mask.shape[0], -1, 2))
        metrics["lv_prey_predator_paired_mask"] = bool(np.all(paired[:, :, 0] == paired[:, :, 1]))
    if problem in {"oup", "lotka_volterra"}:
        high = score >= np.quantile(score, 0.8)
        metrics[f"{problem}_high_state_missing_rate"] = float(np.mean((~mask.reshape(score.shape))[high]))
    return metrics


def safe_ratio(numerator: Any, denominator: Any) -> float:
    denominator = float(denominator)
    if denominator == 0.0:
        return float("nan")
    return float(numerator) / denominator


def safe_quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


def plot_missing_by_feature(missing_by_feature: np.ndarray, output_path: Path, problem: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(np.arange(missing_by_feature.size), missing_by_feature, linewidth=1.5)
    ax.set_xlabel("time" if problem in {"oup", "lotka_volterra"} else "feature")
    ax.set_ylabel("missing fraction")
    ax.set_ylim(-0.02, min(1.02, max(0.1, float(np.max(missing_by_feature)) + 0.1)))
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_observed_missing_hist(observed: np.ndarray, missing: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    if observed.size:
        ax.hist(observed, bins=50, density=True, alpha=0.55, label="observed")
    if missing.size:
        ax.hist(missing, bins=50, density=True, alpha=0.55, label="missing")
    ax.set_xlabel("x_full value")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_score_missing_curve(score_flat: np.ndarray, missing_flat: np.ndarray, output_path: Path) -> None:
    bins = np.quantile(score_flat, np.linspace(0.0, 1.0, 11))
    bins = np.unique(bins)
    centers = []
    rates = []
    if bins.size >= 2:
        for lo, hi in zip(bins[:-1], bins[1:], strict=False):
            in_bin = (score_flat >= lo) & (score_flat <= hi)
            if np.any(in_bin):
                centers.append(float((lo + hi) / 2.0))
                rates.append(float(np.mean(missing_flat[in_bin])))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(centers, rates, marker="o", linewidth=1.5)
    ax.set_xlabel("information score bin")
    ax.set_ylabel("missing fraction")
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_ready(val) for key, val in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


if __name__ == "__main__":
    main()
