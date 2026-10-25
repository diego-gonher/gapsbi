"""Trace and marginal diagnostics for persisted LV/MNAR joint-MCMC pilots."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = (r"$\alpha$", r"$\beta$", r"$\gamma$", r"$\delta$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    chains = [np.load(path)["log_theta"] for path in sorted(args.directory.glob("chain_*.npz"))]
    if not chains:
        raise FileNotFoundError("No persisted chain files found.")
    figure, axes = plt.subplots(4, 2, figsize=(9, 9), constrained_layout=True)
    colors = plt.cm.tab10(np.arange(len(chains)))
    for parameter, label in enumerate(LABELS):
        trace, density = axes[parameter]
        for index, chain in enumerate(chains):
            values = chain[:, parameter]
            trace.plot(values, color=colors[index], linewidth=.45, alpha=.8, label=f"chain {index}" if parameter == 0 else None)
            density.hist(values, bins=55, density=True, histtype="step", color=colors[index], linewidth=1.0)
        trace.set_ylabel(f"log {label}")
        density.set_xlabel(f"log {label}")
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)
    axes[-1, 0].set_xlabel("retained iteration")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".png"), dpi=200)
    figure.savefig(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
