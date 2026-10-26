#!/usr/bin/env python3
"""Create a consolidated 4x3 appendix figure from saved reference analysis CSVs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/analysis/full_vs_partial_references"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=OUT)
    args = parser.parse_args()
    output = args.analysis_dir
    data = pd.read_csv(output / "aggregate_metrics.csv")
    problems = (("oup", "OUP"), ("glm", "GLM"), ("glu", "GLU"), ("lotka_volterra", "Lotka–Volterra"))
    metrics = (("c2st_accuracy", "C2ST accuracy", .5, False),
               ("posterior_mean_shift", "Posterior mean shift", 0., False),
               ("covariance_trace_ratio", "Covariance-trace ratio\n(partial / full)", 1., True))
    fig, axes = plt.subplots(4, 3, figsize=(12, 12), sharex="col")
    fig.subplots_adjust(left=.09, right=.985, bottom=.075, top=.91, hspace=.08, wspace=.15)
    blue = "#1f77b4"
    for row, (problem, display) in enumerate(problems):
        for col, (metric, title, baseline, logarithmic) in enumerate(metrics):
            axis = axes[row, col]
            subset = data[data.problem.eq(problem)]
            for mechanism, linestyle, alpha, marker, legend in (("mcar", "-", 1., "o", "MCAR"), ("mnar", "--", .55, "s", "MNAR")):
                block = subset[subset.mechanism.eq(mechanism)].sort_values("missing_fraction")
                x, y = 100 * block.missing_fraction.to_numpy(), block[metric].to_numpy(float)
                low, high = block[f"{metric}_bootstrap_ci_low"].to_numpy(float), block[f"{metric}_bootstrap_ci_high"].to_numpy(float)
                axis.errorbar(x, y, yerr=np.vstack((y-low, high-y)), color=blue, alpha=alpha, linestyle=linestyle,
                              linewidth=2.0, marker=marker, markersize=6, markerfacecolor=(blue if mechanism == "mcar" else "white"),
                              markeredgewidth=1.5, capsize=3, label=legend)
            axis.axhline(baseline, color="0.35", lw=1.1, ls=":", zorder=0)
            axis.grid(axis="y", alpha=.22, linewidth=.8)
            axis.set_xticks([10, 25, 50])
            if logarithmic:
                axis.set_yscale("log")
                lower, upper = axis.get_ylim()
                candidates = np.array([.1, .2, .3, .5, .7, .9, 1., 1.5, 2., 3., 5., 10., 20., 30., 50., 100.])
                ticks = candidates[(candidates >= lower) & (candidates <= upper)]
                axis.yaxis.set_major_locator(FixedLocator(ticks))
                axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
                axis.yaxis.set_minor_locator(NullLocator())
            if row == 0: axis.set_title(title, fontsize=13, fontweight="bold", pad=12)
            if col == 0:
                axis.set_ylabel(display, fontsize=12, fontweight="bold", labelpad=13)
            if row == len(problems)-1: axis.set_xlabel("")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, 1.02), fontsize=11)
    fig.supxlabel("Missingness fraction (%)", fontsize=12, fontweight="bold", y=.018)
    for ext in ("pdf", "png"):
        fig.savefig(output / "figures" / f"full_vs_partial_reference_appendix_4x3.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__": main()
