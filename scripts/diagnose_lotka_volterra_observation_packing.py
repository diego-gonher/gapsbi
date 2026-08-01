from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gapsbi.masks import (
    LotkaVolterraLogTotalMNARMask,
    LotkaVolterraTimeBlockMCARMask,
    LotkaVolterraTimeMARMask,
)
from gapsbi.simulators import LotkaVolterraSimulator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check Lotka-Volterra observation packing and time-slice masks."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dataset_generation_diagnostics/lotka_volterra"),
    )
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--missing-fraction", type=float, default=0.25)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=24_680)
    args = parser.parse_args()

    simulator = LotkaVolterraSimulator()
    rng = np.random.default_rng(args.seed)
    theta = simulator.sample_theta(args.num_samples, rng)
    x_full = simulator.simulate(theta, rng)
    mechanisms = {
        "time_block_mcar": LotkaVolterraTimeBlockMCARMask(
            missing_fraction=args.missing_fraction,
            block_size=args.block_size,
        ),
        "time_mar_increasing": LotkaVolterraTimeMARMask(
            missing_fraction=args.missing_fraction,
            mode="increasing",
        ),
        "log_total_mnar": LotkaVolterraLogTotalMNARMask(
            missing_fraction=args.missing_fraction,
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = {}
    for name, mask_generator in mechanisms.items():
        mask = mask_generator.generate(x_full, theta, rng)
        diagnostics[name] = summarize_mask(simulator, x_full, mask)
        plot_examples(
            simulator=simulator,
            x_full=x_full,
            mask=mask,
            title=name,
            output_path=args.output_dir / f"lotka_volterra_{name}_packing_examples.png",
        )

    plot_missingness_profiles(
        simulator=simulator,
        diagnostics=diagnostics,
        output_path=args.output_dir / "lotka_volterra_mask_missingness_profiles.png",
    )
    report = build_report(simulator, diagnostics)
    report_path = args.output_dir / "lotka_volterra_observation_packing_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved: {report_path}")


def summarize_mask(
    simulator: LotkaVolterraSimulator,
    x_full: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | np.ndarray]:
    x = x_full.reshape((x_full.shape[0], simulator.num_timepoints, 2))
    mask_time = mask.reshape((mask.shape[0], simulator.num_timepoints, 2))
    paired = mask_time[:, :, 0] == mask_time[:, :, 1]
    time_observed = mask_time[:, :, 0].astype(bool)
    missing = ~time_observed
    log_total = np.log(np.sum(x, axis=2))
    time = np.linspace(0.0, 1.0, simulator.num_timepoints)
    missing_rate_by_time = np.mean(missing, axis=0)
    flat_missing = missing.reshape(-1).astype(float)
    flat_log_total = log_total.reshape(-1)
    repeated_time = np.tile(time, x.shape[0])
    return {
        "paired_consistency": float(np.mean(paired)),
        "element_missing_fraction": float(np.mean(mask == 0)),
        "time_slice_missing_fraction": float(np.mean(missing)),
        "missing_rate_by_time": missing_rate_by_time,
        "time_missing_correlation": safe_corr(repeated_time, flat_missing),
        "log_total_missing_correlation": safe_corr(flat_log_total, flat_missing),
        "mean_log_total_observed": float(np.mean(flat_log_total[flat_missing == 0.0])),
        "mean_log_total_missing": float(np.mean(flat_log_total[flat_missing == 1.0])),
    }


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def build_report(
    simulator: LotkaVolterraSimulator,
    diagnostics: dict[str, dict[str, float | np.ndarray]],
) -> str:
    lines = [
        "Lotka-Volterra observation packing and mask diagnostics",
        "",
        "Observation contract:",
        "  x = [prey_t0, predator_t0, prey_t1, predator_t1, ..., prey_t49, predator_t49]",
        f"  num_timepoints: {simulator.num_timepoints}",
        f"  x_dim: {2 * simulator.num_timepoints}",
        "  mask_unit: time_slice for all LV mechanisms",
        "",
        "Mask diagnostics:",
    ]
    for name, stats in diagnostics.items():
        lines.extend(
            [
                f"  {name}:",
                f"    paired_consistency: {stats['paired_consistency']:.6g}",
                f"    element_missing_fraction: {stats['element_missing_fraction']:.6g}",
                f"    time_slice_missing_fraction: {stats['time_slice_missing_fraction']:.6g}",
                f"    time_missing_correlation: {stats['time_missing_correlation']:.6g}",
                f"    log_total_missing_correlation: {stats['log_total_missing_correlation']:.6g}",
                f"    mean_log_total_observed: {stats['mean_log_total_observed']:.6g}",
                f"    mean_log_total_missing: {stats['mean_log_total_missing']:.6g}",
            ]
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "  paired_consistency must be 1.0; otherwise prey and predator are being",
            "  masked independently, which violates the LV timestamp-level missingness",
            "  contract. MAR should show time dependence. MNAR should show higher",
            "  missingness at larger log total population.",
        ]
    )
    return "\n".join(lines) + "\n"


def plot_missingness_profiles(
    *,
    simulator: LotkaVolterraSimulator,
    diagnostics: dict[str, dict[str, float | np.ndarray]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for name, stats in diagnostics.items():
        ax.plot(
            simulator.timepoints,
            np.asarray(stats["missing_rate_by_time"], dtype=float),
            marker="o",
            markersize=2.5,
            linewidth=1.4,
            label=name,
        )
    ax.set_xlabel("time")
    ax.set_ylabel("missing rate")
    ax.set_title("Lotka-Volterra timestamp missingness profiles")
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_examples(
    *,
    simulator: LotkaVolterraSimulator,
    x_full: np.ndarray,
    mask: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    x = x_full.reshape((x_full.shape[0], simulator.num_timepoints, 2))
    mask_time = mask.reshape((mask.shape[0], simulator.num_timepoints, 2))
    indices = np.arange(min(3, x.shape[0]))
    fig, axes = plt.subplots(len(indices), 1, figsize=(8.0, 3.0 * len(indices)), squeeze=False)
    colors = ("C0", "C2")
    labels = ("prey", "predator")
    for ax, index in zip(axes.ravel(), indices, strict=True):
        for population in range(2):
            observed = mask_time[index, :, population].astype(bool)
            ax.plot(
                simulator.timepoints,
                x[index, :, population],
                color=colors[population],
                linewidth=1.2,
                alpha=0.45,
                label=f"{labels[population]} full" if index == indices[0] else None,
            )
            ax.scatter(
                simulator.timepoints[observed],
                x[index, observed, population],
                color=colors[population],
                s=16,
                label=f"{labels[population]} observed" if index == indices[0] else None,
            )
        missing_times = ~mask_time[index, :, 0].astype(bool)
        for time in simulator.timepoints[missing_times]:
            ax.axvline(time, color="0.85", linewidth=0.6, zorder=0)
        ax.set_yscale("log")
        ax.set_title(f"{title} example {index}")
        ax.set_xlabel("time")
        ax.set_ylabel("population")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
