#!/usr/bin/env python3
"""Compatibility implementation for the canonical reference-target analysis.

Public workflows should invoke ``analyze_full_vs_partial_references.py``. This
module retains its filename for reproducibility and supplies that entry point's
single corrected implementation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gapsbi.analysis.full_vs_partial_references import (
    bootstrap_interval,
    cross_sample_duplicate_summary,
    matched_reference_indices,
)
from gapsbi.evaluation.reference_metrics import FINAL_REFERENCE_AGGREGATION, compute_reference_metrics
from gapsbi.references import (
    load_partial_data_reference_posteriors_hdf5,
    load_reference_posteriors_hdf5,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/analysis/full_vs_partial_references"
# A fresh public-facing location; the versioned v1/v2 artifacts remain immutable.
OUT = ROOT / "outputs/analysis/reference_target_comparison"
INDEPENDENT = ROOT / "references/reference_posteriors_v1/glm_references_independent_replica_v1.h5"
FRACTIONS = (.10, .25, .50)
METRIC_SETTINGS = dict(seed=91357, max_samples_per_observation=10_000, c2st_max_samples_per_observation=2_000,
                       c2st_test_size=None, c2st_n_folds=3, c2st_hidden_layer_scale=5, c2st_max_iter=10_000)
BOOTSTRAP_DRAWS = 2_000


def encode(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=encode), encoding="utf-8")
    temp.replace(path)


def partial_path(fraction: float) -> Path:
    return ROOT / "references/partial_data_references_v1" / f"glm_mnar_eps{round(100*fraction):03d}_references.h5"


def make_figure(aggregate: pd.DataFrame, metric: str, filename: str, ylabel: str, baseline: float, log: bool = False) -> None:
    order = ["oup", "glm", "glu", "lotka_volterra"]
    labels = {"oup":"OUP", "glm":"GLM", "glu":"GLU", "lotka_volterra":"Lotka–Volterra"}
    figure, axes = plt.subplots(4, 1, figsize=(6.5, 10.5), sharex=True, constrained_layout=True)
    color = "#1f77b4"
    for axis, problem in zip(axes, order, strict=True):
        subset = aggregate[aggregate.problem == problem]
        for mechanism, style, alpha, marker, label in (("mcar", "-", 1., "o", "MCAR"), ("mnar", "--", .55, "s", "MNAR")):
            data = subset[subset.mechanism == mechanism].sort_values("missing_fraction")
            x = 100 * data.missing_fraction.to_numpy(); y = data[metric].to_numpy(float)
            lo = data[f"{metric}_bootstrap_ci_low"].to_numpy(float); hi = data[f"{metric}_bootstrap_ci_high"].to_numpy(float)
            axis.errorbar(x, y, yerr=np.vstack((y-lo, hi-y)), color=color, alpha=alpha, linestyle=style, marker=marker,
                          markerfacecolor=(color if mechanism == "mcar" else "white"), markeredgecolor=color, capsize=3, label=label)
        axis.axhline(baseline, color="0.35", linewidth=1, linestyle=":")
        if log: axis.set_yscale("log")
        axis.set_ylabel(ylabel); axis.set_title(labels[problem], loc="left", fontsize=11); axis.grid(axis="y", alpha=.25)
    axes[-1].set_xlabel("Nominal missingness fraction (%)"); axes[-1].set_xticks([10,25,50])
    handles, labels_ = axes[0].get_legend_handles_labels(); figure.legend(handles, labels_, loc="upper center", ncols=2, frameon=False)
    if metric == "c2st_accuracy": axes[0].set_ylim(.45, 1.02)
    for extension in ("pdf", "png"):
        figure.savefig(OUT / "figures" / f"{filename}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def aggregate_glm(rows: list[dict[str, Any]], fraction: float) -> dict[str, Any]:
    result: dict[str, Any] = {"problem": "glm", "mechanism": "mnar", "missing_fraction": fraction,
                              "num_matched_references": len(rows), "original_reference_indices": np.arange(len(rows))}
    for offset, (metric, statistic) in enumerate(FINAL_REFERENCE_AGGREGATION.items()):
        point, low, high, std = bootstrap_interval(np.asarray([row[metric] for row in rows]), statistic=statistic,
                                                    seed=METRIC_SETTINGS["seed"] + offset + round(fraction * 100) * 10,
                                                    draws=BOOTSTRAP_DRAWS)
        result.update({metric: point, f"{metric}_bootstrap_ci_low": low, f"{metric}_bootstrap_ci_high": high,
                       f"{metric}_reference_sd": std, f"{metric}_aggregation": statistic})
    return result


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--independent-full", type=Path, default=INDEPENDENT)
    parser.add_argument("--classifier-seeds", type=int, nargs="*", default=[91357, 91358, 91359])
    args = parser.parse_args()
    OUT = args.out
    if OUT.exists():
        raise FileExistsError(f"{OUT} already exists; version outputs rather than overwrite them.")
    OUT.mkdir(parents=True); (OUT / "figures").mkdir()
    base_level = pd.read_csv(args.base / "reference_level_metrics.csv")
    base_aggregate = pd.read_csv(args.base / "aggregate_metrics.csv")
    base_manifest = json.loads((args.base / "analysis_manifest.json").read_text())
    full_obs, new_full, full_meta, full_diag = load_reference_posteriors_hdf5(args.independent_full)
    if not full_meta.get("independent_replica"):
        raise ValueError("The supplied full reference is not labelled as an independent replica.")
    all_rows, all_null, seed_rows, aggregates, duplicate_rows = [], [], [], [], []
    for fraction in FRACTIONS:
        ppath = partial_path(fraction)
        partial_obs, partial, partial_meta, _ = load_partial_data_reference_posteriors_hdf5(ppath)
        indices = matched_reference_indices(full_obs, partial_obs)
        if not np.array_equal(indices, np.arange(10)):
            raise ValueError("GLM/MNAR must retain all ten original reference IDs.")
        full = new_full[indices]
        duplicate = cross_sample_duplicate_summary(full.reshape(-1, 10), partial.reshape(-1, 10))
        if duplicate["shared_unique_rows"]:
            raise RuntimeError(f"{fraction}: shared full/partial rows remain: {duplicate}")
        rows = compute_reference_metrics(partial, full, progress=True, **METRIC_SETTINGS)
        for i, row in enumerate(rows):
            row.update({"problem": "glm", "mechanism": "mnar", "missing_fraction": fraction,
                        "reference_index": int(indices[i]), "partial_row_index": i,
                        "full_reference_path": str(args.independent_full), "partial_reference_path": str(ppath),
                        "full_num_samples": int(full.shape[1]), "partial_num_samples": int(partial.shape[1]),
                        "theta_dim": int(full.shape[2]), "cross_sample_duplicate_summary": json.dumps(duplicate, sort_keys=True)})
        all_rows.extend(rows); aggregates.append(aggregate_glm(rows, fraction)); duplicate_rows.append({"missing_fraction": fraction, **duplicate})

        old_obs, old, _, _ = load_reference_posteriors_hdf5(ROOT / "references/reference_posteriors_v1/glm_references.h5")
        if not np.array_equal(old_obs["x_full"], full_obs["x_full"]):
            raise ValueError("Canonical and independent full observations differ.")
        null_rows = compute_reference_metrics(old, full, progress=False, **METRIC_SETTINGS)
        for i, row in enumerate(null_rows):
            row.update({"problem": "glm", "mechanism": "mnar", "missing_fraction": fraction,
                        "reference_index": i, "control": "canonical_full_vs_independent_full",
                        "old_full_reference_path": "references/reference_posteriors_v1/glm_references.h5",
                        "independent_full_reference_path": str(args.independent_full)})
        all_null.extend(null_rows)

        for classifier_seed in args.classifier_seeds:
            metrics = dict(METRIC_SETTINGS); metrics["seed"] = int(classifier_seed)
            sensitivity = compute_reference_metrics(partial, full, progress=False, **metrics)
            for i, row in enumerate(sensitivity):
                seed_rows.append({"missing_fraction": fraction, "reference_index": i, "classifier_seed": int(classifier_seed),
                                  "c2st_accuracy": row["c2st_accuracy"]})

    replacement = pd.DataFrame(all_rows)
    retained = base_level[~((base_level.problem == "glm") & (base_level.mechanism == "mnar"))].copy()
    level = pd.concat([retained, replacement], ignore_index=True, sort=False)
    if len(level) != 238 or len(replacement) != 30:
        raise RuntimeError(f"Expected 238 rows with 30 corrected GLM/MNAR rows, got {len(level)} and {len(replacement)}.")
    aggregate = base_aggregate[~((base_aggregate.problem == "glm") & (base_aggregate.mechanism == "mnar"))].copy()
    aggregate = pd.concat([aggregate, pd.DataFrame(aggregates)], ignore_index=True, sort=False)
    if len(aggregate) != len(base_aggregate):
        raise RuntimeError("Aggregate configuration count changed unexpectedly.")
    level.to_csv(OUT / "reference_level_metrics.csv", index=False)
    aggregate.to_csv(OUT / "aggregate_metrics.csv", index=False)
    pd.DataFrame(all_null).to_csv(OUT / "glm_mnar_independent_full_null_control.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(OUT / "glm_mnar_c2st_classifier_seed_sensitivity.csv", index=False)
    pd.DataFrame(duplicate_rows).to_csv(OUT / "glm_mnar_duplicate_preflight.csv", index=False)
    # Preserve the historical generic null controls separately and add the new,
    # per-reference control needed to interpret the corrected GLM/MNAR metrics.
    historical_null = pd.read_csv(args.base / "null_control_metrics.csv")
    pd.concat([historical_null, pd.DataFrame(all_null)], ignore_index=True, sort=False).to_csv(OUT / "null_control_metrics.csv", index=False)
    make_figure(aggregate, "c2st_accuracy", "full_vs_partial_reference_c2st", "C2ST accuracy", .5)
    make_figure(aggregate, "posterior_mean_shift", "full_vs_partial_reference_mean_shift", "Posterior mean shift", 0.)
    make_figure(aggregate, "covariance_trace_ratio", "full_vs_partial_reference_covariance_trace_ratio", "Trace ratio: partial / full", 1., log=True)
    manifest = dict(base_manifest)
    manifest.update({"schema": "gapsbi_full_vs_partial_reference_analysis_v2", "matched_pairs": int(len(level)),
                     "glm_mnar_correction": {"independent_full_reference_path": str(args.independent_full),
                                             "independent_full_metadata": full_meta,
                                             "duplicate_preflight": duplicate_rows,
                                             "classifier_seeds": args.classifier_seeds,
                                             "null_control": "canonical_full_vs_independent_full"},
                     "preserved_base_analysis": str(args.base),
                     "limitations": list(base_manifest.get("limitations", [])) + ["GLM/MNAR C2ST uses an independent full-reference MCMC replica; moment differences include independent Monte Carlo variation."]})
    atomic_json(OUT / "analysis_manifest.json", manifest)
    print(f"Completed canonical corrected reference-target analysis: {OUT}")


if __name__ == "__main__":
    main()
