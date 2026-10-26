#!/usr/bin/env python3
"""Directly compare immutable full-data and partial-data reference posteriors."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gapsbi.analysis.full_vs_partial_references import bootstrap_interval, matched_reference_indices
from gapsbi.evaluation.reference_metrics import compute_reference_metrics
from gapsbi.references import load_partial_data_reference_posteriors_hdf5, load_reference_posteriors_hdf5

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/analysis/full_vs_partial_references"
PROBLEMS = ("oup", "glm", "glu", "lotka_volterra")
MECHANISMS = ("mcar", "mnar")
FRACTIONS = (.10, .25, .50)
METRIC_SETTINGS = dict(seed=91357, max_samples_per_observation=10_000, c2st_max_samples_per_observation=2_000,
                       c2st_test_size=None, c2st_n_folds=3, c2st_hidden_layer_scale=5, c2st_max_iter=10_000)
BOOTSTRAP_DRAWS = 2_000


def encode(value):
    if isinstance(value, (np.integer, np.floating)): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=encode), encoding="utf-8")
    temp.replace(path)


def paths(problem: str, mechanism: str, fraction: float) -> tuple[Path, Path]:
    full = ROOT / "references/reference_posteriors_v1" / f"{problem}_references.h5"
    eps = f"eps{round(100 * fraction):03d}"
    suffix = "_successful_references" if problem == "lotka_volterra" and mechanism == "mnar" and fraction == .50 else "_references"
    partial = ROOT / "references/partial_data_references_v1" / f"{problem}_{mechanism}_{eps}{suffix}.h5"
    return full, partial


def load_pair(problem: str, mechanism: str, fraction: float):
    full_path, partial_path = paths(problem, mechanism, fraction)
    full_obs, full_samples, full_meta, _ = load_reference_posteriors_hdf5(full_path)
    partial_obs, partial_samples, partial_meta, _ = load_partial_data_reference_posteriors_hdf5(partial_path)
    indices = matched_reference_indices(full_obs, partial_obs)
    if full_meta.get("problem") != problem or partial_meta.get("problem") != problem:
        raise ValueError(f"Problem provenance mismatch for {problem}/{mechanism}/{fraction}.")
    if full_meta.get("theta_scaled") or partial_meta.get("theta_scaled"):
        raise ValueError("Reference comparison requires raw parameter samples.")
    if full_samples.shape[2] != partial_samples.shape[2]:
        raise ValueError("Full and partial theta dimensions differ.")
    if not np.all(np.isfinite(full_samples[indices])) or not np.all(np.isfinite(partial_samples)):
        raise ValueError("Reference samples contain non-finite values.")
    return full_path, partial_path, full_obs, full_samples[indices], partial_obs, partial_samples, indices, full_meta, partial_meta


def analyze_configuration(problem: str, mechanism: str, fraction: float) -> tuple[list[dict], dict, dict]:
    full_path, partial_path, full_obs, full, partial_obs, partial, indices, full_meta, partial_meta = load_pair(problem, mechanism, fraction)
    # Orientation is intentional: compute_reference_metrics treats its first input
    # as the estimator, hence it yields Cov(partial)/Cov(full).
    rows = compute_reference_metrics(partial, full, progress=True, **METRIC_SETTINGS)
    for local, row in enumerate(rows):
        original = int(indices[local]); row.update({"problem": problem, "mechanism": mechanism, "missing_fraction": fraction,
                                                     "reference_index": original, "partial_row_index": local,
                                                     "full_reference_path": str(full_path), "partial_reference_path": str(partial_path),
                                                     "full_num_samples": int(full.shape[1]), "partial_num_samples": int(partial.shape[1]),
                                                     "theta_dim": int(full.shape[2])})
    aggregate = {"problem": problem, "mechanism": mechanism, "missing_fraction": fraction, "num_matched_references": len(rows),
                 "original_reference_indices": indices}
    policies = {"c2st_accuracy": "mean", "posterior_mean_shift": "median", "covariance_trace_ratio": "median"}
    for offset, (metric, statistic) in enumerate(policies.items()):
        point, low, high, std = bootstrap_interval(np.asarray([r[metric] for r in rows]), statistic=statistic,
                                                    seed=METRIC_SETTINGS["seed"] + offset + round(fraction * 100) * 10, draws=BOOTSTRAP_DRAWS)
        aggregate.update({metric: point, f"{metric}_bootstrap_ci_low": low, f"{metric}_bootstrap_ci_high": high,
                          f"{metric}_reference_sd": std, f"{metric}_aggregation": statistic})
    provenance = {"full_reference_path": full_path, "partial_reference_path": partial_path, "full_metadata": full_meta,
                  "partial_metadata": partial_meta, "matched_original_reference_indices": indices}
    return rows, aggregate, provenance


def null_control() -> list[dict]:
    rows = []
    for problem in PROBLEMS:
        path = ROOT / "references/reference_posteriors_v1" / f"{problem}_references.h5"
        _, samples, _, _ = load_reference_posteriors_hdf5(path)
        # Independent non-overlapping halves of the same posterior, reference 0.
        null_rows = compute_reference_metrics(samples[:1, :5_000], samples[:1, 5_000:], progress=False, **METRIC_SETTINGS)
        rows.append({"problem": problem, **null_rows[0], "control": "nonoverlapping_halves_same_full_reference", "reference_index": 0})
    return rows


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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)
    levels, aggregates, provenance = [], [], []
    for problem in PROBLEMS:
        for mechanism in MECHANISMS:
            for fraction in FRACTIONS:
                rows, summary, source = analyze_configuration(problem, mechanism, fraction)
                levels.extend(rows); aggregates.append(summary); provenance.append(source)
    level_frame = pd.DataFrame(levels); aggregate_frame = pd.DataFrame(aggregates)
    level_frame.to_csv(OUT / "reference_level_metrics.csv", index=False)
    aggregate_frame.to_csv(OUT / "aggregate_metrics.csv", index=False)
    controls = null_control(); pd.DataFrame(controls).to_csv(OUT / "null_control_metrics.csv", index=False)
    make_figure(aggregate_frame, "c2st_accuracy", "full_vs_partial_reference_c2st", "C2ST accuracy", .5)
    make_figure(aggregate_frame, "posterior_mean_shift", "full_vs_partial_reference_mean_shift", "Posterior mean shift", 0.)
    make_figure(aggregate_frame, "covariance_trace_ratio", "full_vs_partial_reference_covariance_trace_ratio", "Trace ratio: partial / full", 1., log=True)
    atomic_json(OUT / "analysis_manifest.json", {"schema":"gapsbi_full_vs_partial_reference_analysis_v1", "expected_matched_pairs":238,
        "matched_pairs":len(levels), "metric_settings":METRIC_SETTINGS, "bootstrap":{"draws":BOOTSTRAP_DRAWS,"confidence":.95,"unit":"reference_observation"},
        "aggregation":{"c2st_accuracy":"mean","posterior_mean_shift":"median","covariance_trace_ratio":"median"},
        "covariance_trace_ratio_orientation":"trace(Cov_partial)/trace(Cov_full)", "sources":provenance,
        "limitations":["LV/MNAR 50% has eight matched references: original IDs 0,1,3,4,5,6,8,9.","C2ST near 0.5 is a finite-sample null baseline, not proof of equality."]})
    print(f"Completed {len(levels)} matched posterior pairs: {OUT}")


if __name__ == "__main__": main()
