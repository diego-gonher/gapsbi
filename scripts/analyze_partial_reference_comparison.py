"""Summarize and plot full- versus partial-reference metrics.

Currently supports any configuration whose original ``summary.json`` and new
``reference_metrics_with_partial_references.json`` files use the established
GapSBI layouts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

from plot_benchmark_summary import (  # Reuse the paper plotting conventions.
    BUDGET_LABELS, BUDGET_ORDER, FRACTION_COLORS, FRACTION_LABELS,
    FRACTION_ORDER, METHOD_LABELS, METHOD_ORDER, METRICS, style_axis,
)

METRIC_FIELDS = {
    "c2st": ("c2st_accuracy", "reference_c2st_accuracy"),
    "mean_shift": ("posterior_mean_shift", "reference_posterior_mean_shift"),
    "covariance_trace_ratio": ("covariance_trace_ratio", "reference_covariance_trace_ratio"),
}
SEEDS = (101, 202, 303, 404, 505)
BUDGET_X = np.asarray([1e3, 1e4, 1e5], dtype=float)
# Display-only symmetric log-space separation. Values and budget identities are unchanged.
REFERENCE_X_FACTORS = {"full": 10 ** -0.012, "partial": 10 ** 0.012}
OBSERVATION_REDUCERS = {
    "c2st": "mean",
    "mean_shift": "median",
    "covariance_trace_ratio": "median",
}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problem", default="glu")
    p.add_argument("--mechanism", default="mcar")
    p.add_argument("--outputs-root", type=Path, default=Path("."))
    p.add_argument("--outdir", type=Path)
    p.add_argument("--combined-mcar-mnar", action="store_true",
                   help="Create two-row MCAR/MNAR comparison figures.")
    return p.parse_args()


def eps_tag(fraction: float) -> str:
    return f"eps{round(fraction * 100):03d}"


def find_summary(root: Path, budget: str, method: str, problem: str, mechanism: str, fraction: float, seed: int) -> Path:
    if problem == "glm":
        stem = f"glm_raw_{mechanism}_{eps_tag(fraction)}"
    elif problem == "lotka_volterra" and mechanism == "mcar":
        stem = f"lotka_volterra_time_block_mcar_{eps_tag(fraction)}"
    elif problem == "lotka_volterra" and mechanism == "mnar":
        stem = f"lotka_volterra_log_total_mnar_{eps_tag(fraction)}"
    else:
        stem = f"{problem}_{mechanism}_{eps_tag(fraction)}"
    return root / f"outputs_{budget}_sim_budget" / method / problem / mechanism / stem / f"seed_{seed}" / "summary.json"


def aggregate_observation_rows(metric: str, field: str, rows: list[dict]) -> tuple[float, int, bool]:
    """Rebuild one seed aggregate from saved observation rows, not saved summaries."""
    values = np.asarray([row.get(field, float("nan")) for row in rows], dtype=float)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return float("nan"), 0, False
    value = float(np.mean(finite) if OBSERVATION_REDUCERS[metric] == "mean" else np.median(finite))
    return value, int(finite.size), bool(finite.size == values.size)


def collect(root: Path, problem: str, mechanism: str) -> pd.DataFrame:
    if (problem, mechanism) == ("lotka_volterra", "mnar"):
        return collect_lv_mnar_partial_only(root)
    rows = []
    missing = []
    for budget in ("low", "mid", "high"):
        for method in METHOD_ORDER[1:]:  # Full-data NPE has no missing-data conditioning.
            for fraction in FRACTION_ORDER:
                for seed in SEEDS:
                    path = find_summary(root, budget, method, problem, mechanism, fraction, seed)
                    if not path.is_file():
                        missing.append(str(path)); continue
                    common = dict(problem=problem, mechanism=mechanism, budget=budget, method=method,
                                  fraction=fraction, seed=seed)
                    comparison = path.with_name("reference_metrics_with_partial_references.json")
                    if comparison.is_file():
                        partial = json.loads(comparison.read_text())
                        for metric, (field, _) in METRIC_FIELDS.items():
                            for target, key in (("full", "full_reference"), ("partial", "partial_reference")):
                                value, n_valid, complete = aggregate_observation_rows(
                                    metric, field, partial[key]["rows"]
                                )
                                rows.append({**common, "metric": metric, "target": target, "value": value,
                                             "configuration_complete": complete,
                                             "n_valid_reference_observations": n_valid,
                                             "source": str(comparison)})
                    else:
                        original = json.loads(path.read_text())
                        for metric, (_, old_key) in METRIC_FIELDS.items():
                            vals = np.asarray(original[old_key], dtype=float)
                            finite = vals[np.isfinite(vals)]
                            complete = bool(finite.size == vals.size)
                            rows.append({**common, "metric": metric, "target": "full",
                                         "value": (float(np.mean(finite)) if metric == "c2st" else float(np.median(finite)))
                                         if finite.size else float("nan"),
                                         "configuration_complete": complete,
                                         "n_valid_reference_observations": int(finite.size),
                                         "source": str(path)})
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} original evaluations; first: {missing[0]}")
    frame = pd.DataFrame(rows)
    return frame


def collect_lv_mnar_partial_only(root: Path) -> pd.DataFrame:
    """Join immutable compiled full aggregates to new sparse partial metrics."""
    compiled = pd.read_csv(root / "outputs/analysis/full_experiment_results.csv")
    compiled = compiled[(compiled.problem == "lotka_volterra") & (compiled.missingness == "mnar")]
    fields = {"c2st": ("reference_c2st_accuracy_mean", "reference_c2st_accuracy_count"),
              "mean_shift": ("reference_posterior_mean_shift_median", "reference_posterior_mean_shift_count"),
              "covariance_trace_ratio": ("reference_covariance_trace_ratio_median", "reference_covariance_trace_ratio_count")}
    rows, missing = [], []
    for budget in ("low", "mid", "high"):
        for method in METHOD_ORDER[1:]:
            for fraction in FRACTION_ORDER:
                for seed in SEEDS:
                    path = find_summary(root, budget, method, "lotka_volterra", "mnar", fraction, seed)
                    partial_path = path.with_name("reference_metrics_partial_reference_only.json")
                    if not partial_path.is_file():
                        missing.append(str(partial_path)); continue
                    original = compiled[(compiled.method == method) & (compiled.budget == f"{budget}_sim_budget") &
                                        (compiled.epsilon == fraction) & (compiled.seed == seed)]
                    if len(original) != 1:
                        raise ValueError(f"Missing immutable compiled full aggregate for {method}/{budget}/{fraction}/{seed}.")
                    original = original.iloc[0]
                    partial = json.loads(partial_path.read_text())
                    common = dict(problem="lotka_volterra", mechanism="mnar", budget=budget, method=method,
                                  fraction=fraction, seed=seed, source=str(partial_path),
                                  partial_original_reference_indices=partial["partial_reference"]["original_reference_indices"],
                                  reference_subset_mismatch=bool(fraction == .5))
                    for metric, (full_field, count_field) in fields.items():
                        value, n_valid, complete = aggregate_observation_rows(metric, METRIC_FIELDS[metric][0], partial["partial_reference"]["rows"])
                        rows.append({**common, "metric": metric, "target": "full", "value": float(original[full_field]),
                                     "configuration_complete": int(original[count_field]) == 10,
                                     "n_valid_reference_observations": int(original[count_field]),
                                     "source": str(root / "outputs/analysis/full_experiment_results.csv")})
                        rows.append({**common, "metric": metric, "target": "partial", "value": value,
                                     "configuration_complete": complete, "n_valid_reference_observations": n_valid})
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} partial-only evaluations; first: {missing[0]}")
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    base = frame.groupby(["metric", "method", "fraction", "budget", "target"], as_index=False).agg(
        mean=("value", "mean"), std=("value", "std"), n_seeds=("value", "count"),
        n_configurations=("value", "size"),
        n_complete_configurations=("configuration_complete", "sum"),
    )
    wide = base.pivot(
        index=["metric", "method", "fraction", "budget"], columns="target",
        values=["mean", "std", "n_seeds", "n_configurations", "n_complete_configurations"],
    )
    wide.columns = ["_".join(c) for c in wide.columns]
    out = wide.reset_index()
    out["difference_partial_minus_full"] = (
        out["mean_partial"] - out["mean_full"]
        if "mean_partial" in out and "mean_full" in out
        else np.nan
    )
    return out


def correlations(frame: pd.DataFrame) -> dict:
    paired = frame.pivot_table(index=["metric", "method", "fraction", "budget", "seed"], columns="target", values="value")
    paired = paired.dropna(subset=["full", "partial"])
    result: dict[str, dict] = {}
    for metric, group in paired.groupby(level="metric"):
        def corr(x: pd.DataFrame) -> dict:
            if len(x) < 2:
                return {"n": len(x), "pearson": float("nan"), "spearman": float("nan")}
            return {"n": len(x), "pearson": float(pearsonr(x["full"], x["partial"]).statistic),
                    "spearman": float(spearmanr(x["full"], x["partial"]).statistic)}
        result[metric] = {"all": corr(group), "by_fraction": {}, "by_budget": {}}
        for fraction, x in group.groupby(level="fraction"):
            result[metric]["by_fraction"][str(fraction)] = corr(x)
        for budget, x in group.groupby(level="budget"):
            result[metric]["by_budget"][budget] = corr(x)
    # Rankings are deliberately configuration-level seed means, not 10-observation pseudo-replicates.
    finite = frame[np.isfinite(frame.value)].copy()
    means = finite.groupby(["metric", "target", "fraction", "budget", "method"], as_index=False).agg(
        value=("value", "mean"), n_contributing_seeds=("seed", "nunique"),
        n_complete_seeds=("configuration_complete", "sum"),
    )
    rankings = []
    for (metric, target, fraction, budget), group in means.groupby(["metric", "target", "fraction", "budget"]):
        if len(group) == len(METHOD_ORDER) - 1:
            # For ratios, distance from calibrated value one is the meaningful loss.
            group = group.copy()
            group["ranking_score"] = (np.abs(np.log(group.value)) if metric == "covariance_trace_ratio" else group.value)
            order = group.sort_values("ranking_score").method.tolist()
            rankings.append(dict(metric=metric, target=target, fraction=fraction, budget=budget,
                                 better_method=order[0], ranking=">".join(order),
                                 ranking_score="absolute_log_ratio" if metric == "covariance_trace_ratio" else "metric",
                                 contributing_seed_counts={row["method"]: int(row["n_contributing_seeds"]) for _, row in group.iterrows()},
                                 complete_seed_counts={row["method"]: int(row["n_complete_seeds"]) for _, row in group.iterrows()},
                                 incomplete=bool((group.n_contributing_seeds < len(SEEDS)).any() or
                                                 (group.n_complete_seeds < len(SEEDS)).any())))
    result["method_rankings"] = rankings

    # Compare full versus partial rankings using the five-seed method means.
    # These are descriptive configuration-level quantities, not pseudo-replicated
    # tests over the ten shared reference observations.
    method_means = means
    rank_frame = method_means.pivot(index=["metric", "fraction", "budget", "method"], columns="target", values="value").dropna()
    rank_comparisons = []
    for (metric, fraction, budget), group in rank_frame.groupby(level=["metric", "fraction", "budget"]):
        group = group.reset_index().sort_values("method")
        if len(group) != len(METHOD_ORDER) - 1:
            continue
        count_rows = means[(means.metric == metric) & (means.fraction == fraction) & (means.budget == budget)]
        full = group["full"].to_numpy()
        partial = group["partial"].to_numpy()
        if metric == "covariance_trace_ratio":
            full, partial = np.abs(np.log(full)), np.abs(np.log(partial))
            score = "absolute_log_ratio"
        else:
            score = "metric"
        full_order = group.iloc[np.argsort(full)].method.tolist()
        partial_order = group.iloc[np.argsort(partial)].method.tolist()
        rank_comparisons.append(dict(
            metric=metric, fraction=float(fraction), budget=budget, ranking_score=score,
            full_ranking=">".join(full_order), partial_ranking=">".join(partial_order),
            top_method_agrees=full_order[0] == partial_order[0],
            full_ranking_equals_partial=full_order == partial_order,
            spearman=float(spearmanr(full, partial).statistic),
            kendall=float(kendalltau(full, partial).statistic),
            incomplete=bool((count_rows.n_contributing_seeds < len(SEEDS)).any() or
                            (count_rows.n_complete_seeds < len(SEEDS)).any()),
        ))
    result["method_rank_comparisons"] = rank_comparisons
    result["method_rank_agreement_summary"] = {
        metric: {
            "n_configurations": len(items),
            "top_method_agreement_fraction": float(np.mean([x["top_method_agrees"] for x in items])),
            "exact_ranking_agreement_fraction": float(np.mean([x["full_ranking_equals_partial"] for x in items])),
            "mean_spearman": float(np.mean([x["spearman"] for x in items])),
            "mean_kendall": float(np.mean([x["kendall"] for x in items])),
        }
        for metric in METRIC_FIELDS
        if (items := [x for x in rank_comparisons if x["metric"] == metric])
    }
    return result


def sampling_failure_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per incomplete estimator configuration; never infer values."""
    columns = ["problem", "mechanism", "method", "budget", "fraction", "seed",
               "failed_reference_indices", "reference_num_sampling_failures",
               "reference_num_sampling_fallbacks", "source"]
    failures = frame[(~frame.configuration_complete) & (frame.target == "partial")].drop_duplicates(
        ["problem", "mechanism", "method", "budget", "fraction", "seed", "source"]
    )
    rows = []
    for _, item in failures.iterrows():
        comparison_path = Path(item.source)
        comparison = json.loads(comparison_path.read_text())
        if comparison.get("schema") == "gapsbi_partial_reference_only_v1":
            rows.append({
                **{key: item[key] for key in ("problem", "mechanism", "method", "budget", "fraction", "seed")},
                "failed_reference_indices": ";".join(map(str, comparison["partial_reference"].get("missing_original_reference_indices", []))),
                "reference_num_sampling_failures": "partial_reference_unavailable",
                "reference_num_sampling_fallbacks": "",
                "source": str(comparison_path),
            })
            continue
        original = json.loads(comparison_path.with_name("summary.json").read_text())
        rows.append({
            **{key: item[key] for key in ("problem", "mechanism", "method", "budget", "fraction", "seed")},
            "failed_reference_indices": ";".join(map(str, comparison.get(
                "sampling_failure_reference_indices", [
                    row["reference_index"] for row in comparison["full_reference"]["rows"]
                    if not row.get("reference_metrics_valid")
                ]
            ))),
            "reference_num_sampling_failures": original.get("reference_num_sampling_failures"),
            "reference_num_sampling_fallbacks": original.get("reference_num_sampling_fallbacks"),
            "source": str(comparison_path),
        })
    return pd.DataFrame(rows, columns=columns)


def _draw_panel(ax: plt.Axes, panel: pd.DataFrame, spec) -> None:
    style_axis(ax, spec, spec.ylim)
    for fraction in FRACTION_ORDER:
        for target, linestyle, open_marker in (("full", "-", False), ("partial", (0, (4, 2.5)), True)):
            data = panel[(panel.fraction == fraction) & (panel.get(f"mean_{target}").notna())].set_index("budget").reindex(["low", "mid", "high"])
            if data.empty or data[f"mean_{target}"].isna().all():
                continue
            ax.errorbar(BUDGET_X * REFERENCE_X_FACTORS[target], data[f"mean_{target}"],
                        yerr=1.96 * data[f"std_{target}"].fillna(0) / np.sqrt(data[f"n_seeds_{target}"].fillna(1)),
                        color=FRACTION_COLORS[fraction], linestyle=linestyle, marker="o", markersize=3.0,
                        markerfacecolor="none" if open_marker else FRACTION_COLORS[fraction], linewidth=1.0,
                        capsize=1.5, elinewidth=.55)
    ax.set_xscale("log")
    ax.set_xlim(7e2, 1.4e5)
    ax.set_xticks(BUDGET_X, [BUDGET_LABELS[f"{b}_sim_budget"] for b in ("low", "mid", "high")])


def _legend_handles():
    from matplotlib.lines import Line2D
    return [*[Line2D([0], [0], color=FRACTION_COLORS[f], marker="o", label=FRACTION_LABELS[f]) for f in FRACTION_ORDER],
            Line2D([0], [0], color="#4a4a4a", linestyle="-", marker="o", label="Full-data reference"),
            Line2D([0], [0], color="#4a4a4a", linestyle=(0, (4, 2.5)), marker="o", markerfacecolor="none", label="Partial-data reference")]


def plot(summary: pd.DataFrame, outdir: Path, problem: str, mechanism: str) -> list[Path]:
    paths = []
    for metric_key, spec in METRICS.items():
        if metric_key not in METRIC_FIELDS: continue
        fig, axes = plt.subplots(1, len(METHOD_ORDER) - 1, figsize=(12.5, 3.0), sharex=True)
        for ax, method in zip(axes, METHOD_ORDER[1:], strict=True):
            style_axis(ax, spec, spec.ylim)
            ax.set_title(METHOD_LABELS[method].replace("\n", " "), fontsize=9.5, fontweight="bold")
            panel = summary[(summary.metric == metric_key) & (summary.method == method)]
            _draw_panel(ax, panel, spec)
        axes[0].set_ylabel(spec.label)
        fig.supxlabel("Simulation budget")
        fig.suptitle(f"{problem.upper()} {mechanism.upper()}: {spec.label} against full and partial references", y=1.04, fontsize=12, fontweight="bold")
        if (problem, mechanism) == ("lotka_volterra", "mnar"):
            fig.text(.5, -.04, "50%: full reference uses 10 observations; partial reference uses 8 (excluding 2, 7). Descriptive, unmatched comparison.", ha="center", fontsize=7.5)
        fig.legend(handles=_legend_handles(), loc="upper center", ncol=5, bbox_to_anchor=(.5, .99), frameon=False, fontsize=8.5)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            path = outdir / f"{problem}_{mechanism}_{metric_key}_full_vs_partial.{ext}"
            fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
            paths.append(path)
        plt.close(fig)
    return paths


def plot_combined(mcar: pd.DataFrame, mnar: pd.DataFrame, outdir: Path, problem: str) -> list[Path]:
    paths = []
    for metric_key, spec in METRICS.items():
        if metric_key not in METRIC_FIELDS:
            continue
        fig, axes = plt.subplots(2, len(METHOD_ORDER) - 1, figsize=(12.5, 5.1), sharex=True)
        for row, (mechanism, summary) in enumerate((("mcar", mcar), ("mnar", mnar))):
            for col, method in enumerate(METHOD_ORDER[1:]):
                ax = axes[row, col]
                _draw_panel(ax, summary[(summary.metric == metric_key) & (summary.method == method)], spec)
                if row == 0:
                    ax.set_title(METHOD_LABELS[method].replace("\n", " "), fontsize=9.5, fontweight="bold")
                if col == 0:
                    ax.set_ylabel(f"{mechanism.upper()}\n{spec.label}")
        fig.supxlabel("Simulation budget")
        fig.suptitle(f"{problem.upper()}: {spec.label} against full and partial references", y=.99, fontsize=12, fontweight="bold")
        fig.legend(handles=_legend_handles(), loc="upper center", ncol=5, bbox_to_anchor=(.5, .95), frameon=False, fontsize=8.5)
        fig.tight_layout(rect=(0, 0, 1, .90))
        for ext in ("pdf", "png"):
            path = outdir / f"{problem}_mcar_mnar_{metric_key}_full_vs_partial.{ext}"
            fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
            paths.append(path)
        plt.close(fig)
    return paths


def main() -> None:
    a = args()
    if a.outdir is None:
        suffix = f"{a.problem}_mcar_mnar" if a.combined_mcar_mnar else f"{a.problem}_{a.mechanism}"
        a.outdir = Path("outputs/analysis/partial_reference_comparison") / suffix
    a.outdir.mkdir(parents=True, exist_ok=True)
    if a.combined_mcar_mnar:
        mcar = summarize(collect(a.outputs_root, a.problem, "mcar"))
        mnar = summarize(collect(a.outputs_root, a.problem, "mnar"))
        plot_combined(mcar, mnar, a.outdir, a.problem)
        pd.concat([mcar.assign(mechanism="mcar"), mnar.assign(mechanism="mnar")]).to_csv(
            a.outdir / "full_vs_partial_summary.csv", index=False
        )
        print(f"combined_summary_rows={len(mcar) + len(mnar)} outdir={a.outdir}")
        return
    frame = collect(a.outputs_root, a.problem, a.mechanism)
    summary = summarize(frame)
    frame.to_csv(a.outdir / "seed_level_metrics.csv", index=False)
    summary.to_csv(a.outdir / "full_vs_partial_summary.csv", index=False)
    summary[[column for column in summary.columns if column.startswith((
        "metric", "method", "fraction", "budget", "mean_", "std_", "n_seeds_",
        "n_configurations_", "n_complete_configurations_"
    ))]].to_csv(a.outdir / "plot_seed_contributions.csv", index=False)
    sampling_failure_summary(frame).to_csv(a.outdir / "sampling_failures.csv", index=False)
    (a.outdir / "correlations_and_rankings.json").write_text(json.dumps(correlations(frame), indent=2))
    plot(summary, a.outdir, a.problem, a.mechanism)
    if (a.problem, a.mechanism) == ("lotka_volterra", "mnar"):
        availability = frame[frame.target == "partial"][["fraction", "budget", "method", "seed", "n_valid_reference_observations", "partial_original_reference_indices", "reference_subset_mismatch"]]
        availability.to_csv(a.outdir / "reference_availability.csv", index=False)
        (a.outdir / "analysis_manifest.json").write_text(json.dumps({
            "full_reference_source": "outputs/analysis/full_experiment_results.csv",
            "full_reference_metrics_recomputed": False,
            "partial_reference_only": True,
            "fraction_050_partial_original_reference_indices": [0, 1, 3, 4, 5, 6, 8, 9],
            "fraction_050_missing_partial_reference_indices": [2, 7],
            "fraction_050_comparison_caveat": "Full metrics use 10 observations; partial metrics use 8. Descriptive unmatched comparison.",
        }, indent=2))
    print(f"seed_rows={len(frame)} summary_rows={len(summary)} outdir={a.outdir}")

if __name__ == "__main__": main()
