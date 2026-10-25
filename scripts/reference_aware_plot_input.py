"""Build non-mutating benchmark plot inputs for full or partial references.

The original enriched experiment CSV remains immutable.  This module overlays
only the three reference-dependent seed-level metrics from the completed
partial-reference analyses.  It deliberately does not touch calibration,
runtime, or Full-Data NPE values.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE_METRICS = {
    "c2st": ("reference_c2st_accuracy_mean", "reference_c2st_accuracy_count"),
    "mean_shift": (
        "reference_posterior_mean_shift_median",
        "reference_posterior_mean_shift_count",
    ),
    "covariance_trace_ratio": (
        "reference_covariance_trace_ratio_median",
        "reference_covariance_trace_ratio_count",
    ),
}
PARTIAL_CAMPAIGNS = (
    ("glu", "mcar"), ("glu", "mnar"),
    ("glm", "mcar"), ("glm", "mnar"),
    ("oup", "mcar"), ("oup", "mnar"),
    ("lotka_volterra", "mcar"), ("lotka_volterra", "mnar"),
)
KEYS = ("problem", "missingness", "epsilon", "method", "budget", "seed")


@dataclass(frozen=True)
class PlotInputs:
    full: pd.DataFrame
    partial: pd.DataFrame
    comparison: pd.DataFrame


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["problem"] = out["problem"].astype(str)
    out["missingness"] = out["missingness"].astype(str)
    out["method"] = out["method"].astype(str)
    out["budget"] = out["budget"].astype(str)
    out["budget"] = out["budget"].replace({
        "low": "low_sim_budget", "mid": "mid_sim_budget", "high": "high_sim_budget",
    })
    out["seed"] = pd.to_numeric(out["seed"], errors="raise").astype(int)
    out["epsilon"] = pd.to_numeric(out["epsilon"], errors="coerce")
    out.loc[out["epsilon"] > 1.0, "epsilon"] /= 100.0
    out["epsilon"] = out["epsilon"].round(2)
    return out


def _assert_unique(df: pd.DataFrame, keys: tuple[str, ...], label: str) -> None:
    duplicates = df.duplicated(list(keys), keep=False)
    if duplicates.any():
        example = df.loc[duplicates, list(keys)].head(3).to_dict("records")
        raise ValueError(f"{label} has non-unique join keys; examples: {example}")


def load_partial_seed_metrics(root: Path) -> pd.DataFrame:
    """Load only partial-target seed metrics from all completed campaigns."""
    frames: list[pd.DataFrame] = []
    expected = set(PARTIAL_CAMPAIGNS)
    for problem, mechanism in PARTIAL_CAMPAIGNS:
        path = root / "outputs/analysis/partial_reference_comparison" / f"{problem}_{mechanism}" / "seed_level_metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed partial-analysis input: {path}")
        frame = pd.read_csv(path)
        required = {"problem", "mechanism", "budget", "method", "fraction", "seed", "metric", "target", "value", "n_valid_reference_observations"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} lacks columns: {sorted(missing)}")
        frame = frame[frame["target"] == "partial"].copy()
        if set(frame[["problem", "mechanism"]].drop_duplicates().itertuples(index=False, name=None)) != {(problem, mechanism)}:
            raise ValueError(f"Unexpected campaign identity in {path}")
        frames.append(frame)
    partial = pd.concat(frames, ignore_index=True)
    if set(partial[["problem", "mechanism"]].drop_duplicates().itertuples(index=False, name=None)) != expected:
        raise ValueError("Partial campaign set is incomplete")
    partial = partial.rename(columns={"mechanism": "missingness", "fraction": "epsilon"})
    partial = _normalize(partial)
    required_metrics = set(REFERENCE_METRICS)
    if set(partial["metric"].unique()) != required_metrics:
        raise ValueError(f"Unexpected partial metrics: {sorted(partial['metric'].unique())}")
    _assert_unique(partial, (*KEYS, "metric"), "Partial seed metrics")
    return partial


def build_plot_inputs(enriched_path: Path, partial_root: Path) -> PlotInputs:
    """Return immutable-full, partial-overlay, and long comparison inputs."""
    full = _normalize(pd.read_csv(enriched_path))
    _assert_unique(full, KEYS, "Enriched benchmark CSV")
    partial_metrics = load_partial_seed_metrics(partial_root)
    partial = full.copy()
    partial["reference_plot_mode"] = "partial"
    partial["partial_reference_provenance"] = ""
    partial["partial_reference_original_indices"] = ""
    partial["partial_reference_subset_mismatch"] = False

    # Only completed MCAR/MNAR campaigns are eligible for overlay.  Historical
    # MAR rows deliberately remain untouched, so the adapter is safe to use
    # against the full historical CSV as well.
    campaign_pairs = set(PARTIAL_CAMPAIGNS)
    eligible = (partial["method"] != "npe_full_data") & partial.apply(
        lambda row: (row["problem"], row["missingness"]) in campaign_pairs, axis=1
    )
    target_keys = partial.loc[eligible, list(KEYS)]
    expected_rows = len(target_keys) * len(REFERENCE_METRICS)
    joined = target_keys.merge(partial_metrics, on=list(KEYS), how="left", validate="one_to_many", indicator=True)
    if len(joined) != expected_rows or (joined["_merge"] != "both").any():
        absent = joined.loc[joined["_merge"] != "both", list(KEYS)].drop_duplicates().head(5).to_dict("records")
        raise ValueError(f"Partial metrics do not strictly cover benchmark configurations; examples: {absent}")

    for metric, (value_column, count_column) in REFERENCE_METRICS.items():
        rows = partial_metrics[partial_metrics["metric"] == metric].copy()
        _assert_unique(rows, KEYS, f"Partial {metric} metrics")
        values = rows.set_index(list(KEYS))["value"]
        counts = rows.set_index(list(KEYS))["n_valid_reference_observations"]
        key_index = pd.MultiIndex.from_frame(partial.loc[eligible, list(KEYS)])
        partial.loc[eligible, value_column] = values.reindex(key_index).to_numpy()
        partial.loc[eligible, count_column] = counts.reindex(key_index).to_numpy()

    # Campaign provenance and sparse original-index metadata are recorded once per seed/config.
    metadata = partial_metrics.drop_duplicates(list(KEYS)).set_index(list(KEYS))
    key_index = pd.MultiIndex.from_frame(partial.loc[eligible, list(KEYS)])
    partial.loc[eligible, "partial_reference_provenance"] = metadata["source"].reindex(key_index).fillna("").to_numpy()
    if "partial_original_reference_indices" in metadata:
        partial.loc[eligible, "partial_reference_original_indices"] = metadata["partial_original_reference_indices"].reindex(key_index).fillna("").to_numpy()
    else:
        partial.loc[eligible, "partial_reference_original_indices"] = "[0,1,2,3,4,5,6,7,8,9]"
    if "reference_subset_mismatch" in metadata:
        partial.loc[eligible, "partial_reference_subset_mismatch"] = metadata["reference_subset_mismatch"].reindex(key_index).fillna(False).astype(bool).to_numpy()

    # Full-data NPE has no missing-data reference target and is unchanged in every mode.
    partial.loc[~eligible, "reference_plot_mode"] = "full_or_historical"
    full = full.copy()
    full["reference_plot_mode"] = "full"
    comparison = pd.concat([full.assign(reference_target="full"), partial.assign(reference_target="partial")], ignore_index=True)
    return PlotInputs(full=full, partial=partial, comparison=comparison)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/analysis/full_experiment_results_with_diagnostics.csv"))
    parser.add_argument("--partial-root", type=Path, default=Path("."))
    parser.add_argument("--reference-mode", choices=("full", "partial", "comparison"), default="partial")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    inputs = build_plot_inputs(args.input, args.partial_root)
    output = getattr(inputs, args.reference_mode)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(f"rows={len(output)} mode={args.reference_mode} output={args.out}")


if __name__ == "__main__":
    main()
