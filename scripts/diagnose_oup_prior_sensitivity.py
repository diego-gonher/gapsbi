from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gapsbi.references import (
    compute_oup_grid_posterior,
    load_reference_posteriors_hdf5,
    sample_oup_grid_posterior,
)
from gapsbi.simulators import OUPSimulator


PRIORS = {
    "current_wide": (np.array([0.0, -2.0]), np.array([2.0, 3.0])),
    "previous_rise": (np.array([0.0, -2.0]), np.array([2.0, 2.0])),
    "centered_y0": (np.array([0.0, np.log(2.0)]), np.array([2.0, np.log(20.0)])),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare OUP prior alternatives using prior predictive and reference posterior diagnostics."
    )
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=Path("references/reference_posteriors_v1/oup_references.h5"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dataset_generation_diagnostics/oup/prior_sensitivity"),
    )
    parser.add_argument("--grid-resolution", type=int, default=800)
    parser.add_argument("--num-prior-samples", type=int, default=10_000)
    parser.add_argument("--num-posterior-samples", type=int, default=10_000)
    parser.add_argument("--plot-samples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations, current_samples, metadata, current_diagnostics = (
        load_reference_posteriors_hdf5(args.reference_path)
    )
    if metadata.get("problem") != "oup":
        raise ValueError(f"Expected OUP references, got {metadata.get('problem')!r}.")

    prior_predictive = compute_prior_predictive_summaries(
        num_samples=args.num_prior_samples,
        seed=args.seed,
    )
    posterior_results = compute_posterior_sensitivity(
        x_full=observations["x_full"],
        current_samples=current_samples,
        current_diagnostics=current_diagnostics,
        grid_resolution=args.grid_resolution,
        num_posterior_samples=args.num_posterior_samples,
        seed=args.seed + 1,
    )

    report_path = args.output_dir / "oup_prior_sensitivity_report.txt"
    report_path.write_text(
        build_report(
            reference_path=args.reference_path,
            prior_predictive=prior_predictive,
            posterior_results=posterior_results,
            theta_true=observations["theta_true"],
            grid_resolution=args.grid_resolution,
        ),
        encoding="utf-8",
    )
    paths = [
        plot_prior_predictive(
            prior_predictive,
            args.output_dir / "oup_prior_sensitivity_prior_predictive.png",
        ),
        plot_posterior_mean_shifts(
            posterior_results,
            observations["theta_true"],
            args.output_dir / "oup_prior_sensitivity_posterior_mean_shifts.png",
        ),
        plot_boundary_mass(
            posterior_results,
            args.output_dir / "oup_prior_sensitivity_boundary_mass.png",
        ),
        plot_posterior_contours(
            posterior_results,
            observations["theta_true"],
            args.output_dir / "oup_prior_sensitivity_posterior_contours.png",
            max_samples=args.plot_samples,
            seed=args.seed + 2,
        ),
    ]

    print(report_path)
    for path in paths:
        print(path)


def compute_prior_predictive_summaries(
    *,
    num_samples: int,
    seed: int,
) -> dict[str, dict[str, np.ndarray | float]]:
    results = {}
    for prior_idx, (name, (prior_low, prior_high)) in enumerate(PRIORS.items()):
        simulator = OUPSimulator(prior_low=prior_low, prior_high=prior_high)
        rng = np.random.default_rng(seed + prior_idx)
        theta = simulator.sample_theta(num_samples, rng)
        x = simulator.simulate(theta, rng)
        theta2 = np.exp(theta[:, 1])
        results[name] = {
            "prior_low": prior_low,
            "prior_high": prior_high,
            "theta2_q05": float(np.quantile(theta2, 0.05)),
            "theta2_median": float(np.quantile(theta2, 0.5)),
            "theta2_q95": float(np.quantile(theta2, 0.95)),
            "fraction_theta2_below_y0": float(np.mean(theta2 < simulator.y0)),
            "fraction_theta2_near_y0": float(
                np.mean((theta2 >= 0.9 * simulator.y0) & (theta2 <= 1.1 * simulator.y0))
            ),
            "fraction_theta2_above_y0": float(np.mean(theta2 > simulator.y0)),
            "x_q05": np.quantile(x, 0.05, axis=0),
            "x_median": np.quantile(x, 0.5, axis=0),
            "x_q95": np.quantile(x, 0.95, axis=0),
            "final_x_q05": float(np.quantile(x[:, -1], 0.05)),
            "final_x_median": float(np.quantile(x[:, -1], 0.5)),
            "final_x_q95": float(np.quantile(x[:, -1], 0.95)),
        }
    return results


def compute_posterior_sensitivity(
    *,
    x_full: np.ndarray,
    current_samples: np.ndarray,
    current_diagnostics: dict[str, np.ndarray],
    grid_resolution: int,
    num_posterior_samples: int,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    results: dict[str, dict[str, np.ndarray]] = {}
    current_mean = np.asarray(current_diagnostics["grid_mean"], dtype=float)
    current_cov = np.asarray(current_diagnostics["grid_cov"], dtype=float)
    current_boundary = np.asarray(current_diagnostics["grid_boundary_mass"], dtype=float)
    results["current_wide"] = posterior_summary_from_samples(
        current_samples,
        mean=current_mean,
        cov=current_cov,
        boundary_mass=current_boundary,
    )

    for name, (prior_low, prior_high) in PRIORS.items():
        if name == "current_wide":
            continue
        simulator = OUPSimulator(prior_low=prior_low, prior_high=prior_high)
        means = []
        covs = []
        boundary_mass = []
        ess = []
        samples = []
        for obs_idx in range(x_full.shape[0]):
            posterior = compute_oup_grid_posterior(
                x_full[obs_idx],
                simulator=simulator,
                grid_resolution=grid_resolution,
            )
            samples_i = sample_oup_grid_posterior(
                posterior,
                num_reference_samples=num_posterior_samples,
                rng=rng,
            )
            means.append(np.asarray(posterior["mean"], dtype=float))
            covs.append(np.asarray(posterior["cov"], dtype=float))
            boundary_mass.append(float(posterior["boundary_mass"]))
            ess.append(float(posterior["ess"]))
            samples.append(samples_i)
        results[name] = posterior_summary_from_samples(
            np.asarray(samples),
            mean=np.asarray(means),
            cov=np.asarray(covs),
            boundary_mass=np.asarray(boundary_mass),
            ess=np.asarray(ess),
        )
    return results


def posterior_summary_from_samples(
    samples: np.ndarray,
    *,
    mean: np.ndarray,
    cov: np.ndarray,
    boundary_mass: np.ndarray,
    ess: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    theta2 = np.exp(samples[:, :, 1])
    std = np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 0.0))
    area = np.sqrt(np.maximum(np.linalg.det(cov), 0.0))
    return {
        "samples": samples,
        "mean": mean,
        "cov": cov,
        "std": std,
        "theta2_median": np.median(theta2, axis=1),
        "theta2_q05": np.quantile(theta2, 0.05, axis=1),
        "theta2_q95": np.quantile(theta2, 0.95, axis=1),
        "boundary_mass": boundary_mass,
        "posterior_area": area,
        "ess": np.full(samples.shape[0], np.nan) if ess is None else ess,
    }


def build_report(
    *,
    reference_path: Path,
    prior_predictive: dict[str, dict[str, np.ndarray | float]],
    posterior_results: dict[str, dict[str, np.ndarray]],
    theta_true: np.ndarray,
    grid_resolution: int,
) -> str:
    current = posterior_results["current_wide"]
    lines = [
        "OUP prior sensitivity diagnostics",
        "=================================",
        f"Reference artifact: {reference_path}",
        f"Grid resolution for alternatives: {grid_resolution} x {grid_resolution}",
        "",
        "Prior predictive summaries:",
    ]
    for name, stats in prior_predictive.items():
        low = np.asarray(stats["prior_low"], dtype=float)
        high = np.asarray(stats["prior_high"], dtype=float)
        lines.extend(
            [
                f"- {name}: theta1 U({low[0]:.3g}, {high[0]:.3g}), "
                f"log_theta2 U({low[1]:.6g}, {high[1]:.6g})",
                f"  theta2 q05/median/q95: {stats['theta2_q05']:.6g}, "
                f"{stats['theta2_median']:.6g}, {stats['theta2_q95']:.6g}",
                f"  P(theta2 < y0): {stats['fraction_theta2_below_y0']:.6f}; "
                f"P(theta2 near y0): {stats['fraction_theta2_near_y0']:.6f}; "
                f"P(theta2 > y0): {stats['fraction_theta2_above_y0']:.6f}",
                f"  final x q05/median/q95: {stats['final_x_q05']:.6g}, "
                f"{stats['final_x_median']:.6g}, {stats['final_x_q95']:.6g}",
            ]
        )

    lines.extend(["", "Posterior comparison to current widened prior:"])
    for name, result in posterior_results.items():
        shift = np.linalg.norm(result["mean"] - current["mean"], axis=1)
        theta2_shift = np.abs(result["theta2_median"] - current["theta2_median"])
        area_ratio = result["posterior_area"] / np.maximum(current["posterior_area"], 1e-15)
        truth_z = np.linalg.norm(
            (theta_true - result["mean"]) / np.maximum(result["std"], 1e-12),
            axis=1,
        )
        lines.extend(
            [
                f"- {name}:",
                f"  mean shift L2 vs current: median={np.median(shift):.6g}, max={np.max(shift):.6g}",
                f"  theta2 median shift vs current: median={np.median(theta2_shift):.6g}, max={np.max(theta2_shift):.6g}",
                f"  boundary mass: median={np.median(result['boundary_mass']):.6g}, max={np.max(result['boundary_mass']):.6g}",
                f"  posterior area ratio vs current: median={np.median(area_ratio):.6g}, max={np.max(area_ratio):.6g}",
                f"  theta_true standardized distance: median={np.median(truth_z):.6g}, max={np.max(truth_z):.6g}",
            ]
        )
    lines.extend(
        [
            "",
            "Critical interpretation:",
            "- Previous-rise checks whether the original RISE upper bound materially changes the regenerated references.",
            "- Centered-y0 tests a qualitatively different prior and excludes low-equilibrium dynamics.",
            "- A better prior should reduce structural prior-predictive asymmetry without creating boundary-truncated or weakly identified posteriors.",
        ]
    )
    return "\n".join(lines) + "\n"


def plot_prior_predictive(
    prior_predictive: dict[str, dict[str, np.ndarray | float]],
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    time = np.arange(len(next(iter(prior_predictive.values()))["x_median"]))
    colors = {
        "current_wide": "C0",
        "previous_rise": "C1",
        "centered_y0": "C2",
    }
    for name, stats in prior_predictive.items():
        color = colors[name]
        ax.plot(time, stats["x_median"], color=color, label=name)
        ax.fill_between(time, stats["x_q05"], stats["x_q95"], color=color, alpha=0.18)
    ax.axhline(10.0, color="black", linestyle="--", linewidth=1.0, label="y0")
    ax.set_xlabel("time index")
    ax.set_ylabel("x")
    ax.set_title("OUP prior predictive 5/50/95% bands")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_posterior_mean_shifts(
    posterior_results: dict[str, dict[str, np.ndarray]],
    theta_true: np.ndarray,
    output_path: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    current = posterior_results["current_wide"]
    obs = np.arange(theta_true.shape[0])
    for name, result in posterior_results.items():
        shift = np.linalg.norm(result["mean"] - current["mean"], axis=1)
        theta2_shift = np.abs(result["theta2_median"] - current["theta2_median"])
        axes[0].plot(obs, shift, marker="o", label=name)
        axes[1].plot(obs, theta2_shift, marker="o", label=name)
    axes[0].set_title("Mean shift vs current")
    axes[0].set_ylabel("L2 shift in [theta1, log_theta2]")
    axes[1].set_title("Equilibrium median shift vs current")
    axes[1].set_ylabel("|Delta median theta2|")
    for ax in axes:
        ax.set_xlabel("observation")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_boundary_mass(
    posterior_results: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    obs = np.arange(next(iter(posterior_results.values()))["boundary_mass"].shape[0])
    for name, result in posterior_results.items():
        ax.plot(obs, result["boundary_mass"], marker="o", label=name)
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.set_xlabel("observation")
    ax.set_ylabel("grid boundary mass")
    ax.set_title("OUP posterior boundary mass under prior alternatives")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_posterior_contours(
    posterior_results: dict[str, dict[str, np.ndarray]],
    theta_true: np.ndarray,
    output_path: Path,
    *,
    max_samples: int,
    seed: int,
) -> Path:
    rng = np.random.default_rng(seed)
    names = list(posterior_results.keys())
    fig, axes = plt.subplots(len(names), theta_true.shape[0], figsize=(20, 7), sharex=False, sharey=False)
    for row, name in enumerate(names):
        samples = posterior_results[name]["samples"]
        if samples.shape[1] > max_samples:
            idx = rng.choice(samples.shape[1], size=max_samples, replace=False)
            samples = samples[:, idx, :]
        for obs_idx in range(theta_true.shape[0]):
            ax = axes[row, obs_idx]
            theta2 = np.exp(samples[obs_idx, :, 1])
            ax.hist2d(samples[obs_idx, :, 0], theta2, bins=70, cmap="Blues")
            ax.scatter(
                theta_true[obs_idx, 0],
                np.exp(theta_true[obs_idx, 1]),
                c="C3",
                marker="x",
                s=35,
            )
            ax.set_title(f"{name}\nobs {obs_idx:02d}", fontsize=8)
            ax.tick_params(axis="both", labelsize=6)
            if obs_idx == 0:
                ax.set_ylabel("theta2", fontsize=8)
            if row == len(names) - 1:
                ax.set_xlabel("theta1", fontsize=8)
    fig.suptitle("OUP posterior samples under prior alternatives", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    main()
