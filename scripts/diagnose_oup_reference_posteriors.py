from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gapsbi.references import load_reference_posteriors_hdf5


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check OUP reference posterior boundary mass and prior-bound geometry."
    )
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=Path("references/reference_posteriors_v1/oup_references.h5"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dataset_generation_diagnostics/oup"),
    )
    parser.add_argument("--max-samples", type=int, default=10_000)
    parser.add_argument("--bins", type=int, default=90)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations, theta_samples, metadata, diagnostics = load_reference_posteriors_hdf5(
        args.reference_path
    )
    if metadata.get("problem") != "oup":
        raise ValueError(f"Expected OUP references, got {metadata.get('problem')!r}.")
    if "grid_boundary_mass" not in diagnostics:
        raise ValueError("OUP reference diagnostics do not contain grid_boundary_mass.")

    prior_low = np.asarray(diagnostics["prior_low"], dtype=float)
    prior_high = np.asarray(diagnostics["prior_high"], dtype=float)
    grid_resolution = int(metadata.get("grid_resolution", 800))
    cell_width = (prior_high - prior_low) / grid_resolution
    theta_true = observations["theta_true"]
    samples = _subsample(theta_samples, args.max_samples, args.seed)

    report_path = args.output_dir / "oup_reference_boundary_mass_report.txt"
    report_path.write_text(
        _build_report(
            reference_path=args.reference_path,
            theta_true=theta_true,
            samples=samples,
            boundary_mass=np.asarray(diagnostics["grid_boundary_mass"], dtype=float),
            prior_low=prior_low,
            prior_high=prior_high,
            cell_width=cell_width,
            y0=float(metadata.get("simulator", {}).get("y0", 10.0)),
        ),
        encoding="utf-8",
    )

    paths = [
        _plot_bounds_overlay(
            output_path=args.output_dir / "oup_reference_contours_log_theta2_bounds.png",
            theta_true=theta_true,
            samples=samples,
            prior_low=prior_low,
            prior_high=prior_high,
            bins=args.bins,
        ),
        _plot_equilibrium_overlay(
            output_path=args.output_dir / "oup_reference_contours_theta2_bounds.png",
            theta_true=theta_true,
            samples=samples,
            prior_low=prior_low,
            prior_high=prior_high,
            bins=args.bins,
            y0=float(metadata.get("simulator", {}).get("y0", 10.0)),
        ),
    ]

    print(report_path)
    for path in paths:
        print(path)


def _subsample(theta_samples: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if theta_samples.shape[1] <= max_samples:
        return theta_samples
    rng = np.random.default_rng(seed)
    idx = rng.choice(theta_samples.shape[1], size=max_samples, replace=False)
    return theta_samples[:, idx, :]


def _build_report(
    *,
    reference_path: Path,
    theta_true: np.ndarray,
    samples: np.ndarray,
    boundary_mass: np.ndarray,
    prior_low: np.ndarray,
    prior_high: np.ndarray,
    cell_width: np.ndarray,
    y0: float,
) -> str:
    theta2_true = np.exp(theta_true[:, 1])
    theta2_samples = np.exp(samples[:, :, 1])
    edge_mass = _sample_edge_mass(samples, prior_low, prior_high, cell_width)
    truth_margin = np.minimum(
        (theta_true - prior_low) / (prior_high - prior_low),
        (prior_high - theta_true) / (prior_high - prior_low),
    ).min(axis=1)

    lines = [
        "OUP reference posterior boundary check",
        "======================================",
        f"Reference artifact: {reference_path}",
        f"Prior bounds: theta1=[{prior_low[0]:.6g}, {prior_high[0]:.6g}], "
        f"log_theta2=[{prior_low[1]:.6g}, {prior_high[1]:.6g}]",
        f"theta2 support: [{np.exp(prior_low[1]):.6g}, {np.exp(prior_high[1]):.6g}]",
        f"Initial state y0: {y0:.6g}",
        "",
        "obs  theta1_true  log_theta2_true  theta2_true  grid_boundary_mass  sample_edge_mass  min_truth_margin",
    ]
    for i in range(theta_true.shape[0]):
        lines.append(
            f"{i:02d}   "
            f"{theta_true[i, 0]:.6f}      "
            f"{theta_true[i, 1]:.6f}         "
            f"{theta2_true[i]:.6f}     "
            f"{boundary_mass[i]:.6e}        "
            f"{edge_mass[i]:.6e}      "
            f"{truth_margin[i]:.6f}"
        )
    lines.extend(
        [
            "",
            f"grid_boundary_mass max: {np.max(boundary_mass):.6e}",
            f"sample_edge_mass max: {np.max(edge_mass):.6e}",
            f"theta2 posterior median range: "
            f"[{np.min(np.median(theta2_samples, axis=1)):.6g}, "
            f"{np.max(np.median(theta2_samples, axis=1)):.6g}]",
            "Interpretation: boundary mass should be small; large values would mean the posterior is being truncated by the chosen prior.",
        ]
    )
    return "\n".join(lines) + "\n"


def _sample_edge_mass(
    samples: np.ndarray,
    prior_low: np.ndarray,
    prior_high: np.ndarray,
    cell_width: np.ndarray,
) -> np.ndarray:
    near_low = samples <= prior_low[None, None, :] + cell_width[None, None, :]
    near_high = samples >= prior_high[None, None, :] - cell_width[None, None, :]
    return np.mean(np.any(near_low | near_high, axis=2), axis=1)


def _plot_bounds_overlay(
    *,
    output_path: Path,
    theta_true: np.ndarray,
    samples: np.ndarray,
    prior_low: np.ndarray,
    prior_high: np.ndarray,
    bins: int,
) -> Path:
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharex=True, sharey=True)
    for obs_idx, ax in enumerate(axes.ravel()):
        _density_contour(
            ax,
            samples[obs_idx, :, 0],
            samples[obs_idx, :, 1],
            xlim=(prior_low[0], prior_high[0]),
            ylim=(prior_low[1], prior_high[1]),
            bins=bins,
        )
        ax.scatter(theta_true[obs_idx, 0], theta_true[obs_idx, 1], c="C3", marker="x", s=45)
        ax.set_title(f"obs {obs_idx:02d}", fontsize=9)
        ax.set_xlim(prior_low[0], prior_high[0])
        ax.set_ylim(prior_low[1], prior_high[1])
        ax.tick_params(axis="both", labelsize=7)
    fig.supxlabel("theta1")
    fig.supylabel("log_theta2")
    fig.suptitle("OUP reference posteriors relative to prior bounds", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_equilibrium_overlay(
    *,
    output_path: Path,
    theta_true: np.ndarray,
    samples: np.ndarray,
    prior_low: np.ndarray,
    prior_high: np.ndarray,
    bins: int,
    y0: float,
) -> Path:
    theta2_samples = np.exp(samples[:, :, 1])
    theta2_true = np.exp(theta_true[:, 1])
    theta2_low = float(np.exp(prior_low[1]))
    theta2_high = float(max(np.exp(prior_high[1]), y0) * 1.03)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharex=True, sharey=True)
    for obs_idx, ax in enumerate(axes.ravel()):
        _density_contour(
            ax,
            samples[obs_idx, :, 0],
            theta2_samples[obs_idx],
            xlim=(prior_low[0], prior_high[0]),
            ylim=(theta2_low, theta2_high),
            bins=bins,
        )
        ax.scatter(theta_true[obs_idx, 0], theta2_true[obs_idx], c="C3", marker="x", s=45)
        ax.axhline(y0, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(f"obs {obs_idx:02d}", fontsize=9)
        ax.set_xlim(prior_low[0], prior_high[0])
        ax.set_ylim(theta2_low, theta2_high)
        ax.tick_params(axis="both", labelsize=7)
    fig.supxlabel("theta1")
    fig.supylabel("theta2 = exp(log_theta2)")
    fig.suptitle("OUP equilibrium posteriors relative to y0 and bounds", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _density_contour(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    bins: int,
) -> None:
    hist, x_edges, y_edges = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[xlim, ylim],
    )
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    ax.imshow(
        hist.T,
        origin="lower",
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        aspect="auto",
        cmap="Blues",
        alpha=0.75,
    )
    levels = _credible_levels(hist, probs=(0.9, 0.5))
    if levels.size:
        ax.contour(
            x_centers,
            y_centers,
            hist.T,
            levels=levels,
            colors=["C0"] * len(levels),
            linewidths=1.1,
        )


def _credible_levels(hist: np.ndarray, probs: tuple[float, ...]) -> np.ndarray:
    flat = np.sort(hist.ravel())[::-1]
    total = flat.sum()
    if total <= 0:
        return np.array([], dtype=float)
    cdf = np.cumsum(flat) / total
    levels = []
    for prob in probs:
        idx = min(np.searchsorted(cdf, prob), flat.size - 1)
        level = flat[idx]
        if level > 0:
            levels.append(level)
    return np.unique(np.sort(levels))


if __name__ == "__main__":
    main()
