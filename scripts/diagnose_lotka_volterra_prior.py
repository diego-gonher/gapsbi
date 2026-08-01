from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gapsbi.simulators import LotkaVolterraSimulator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run prior predictive diagnostics for the Lotka-Volterra task."
    )
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=Path("references/reference_posteriors_v1/lotka_volterra_references.h5"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dataset_generation_diagnostics/lotka_volterra"),
    )
    parser.add_argument("--num-prior-samples", type=int, default=2_000)
    parser.add_argument("--num-plot-trajectories", type=int, default=80)
    parser.add_argument("--seed", type=int, default=98_765)
    args = parser.parse_args()

    simulator, theta_true = load_reference_simulator(args.reference_path)
    rng = np.random.default_rng(args.seed)
    theta = simulator.sample_theta(args.num_prior_samples, rng)
    states, failure_mask = solve_prior_trajectories(simulator, theta)
    report = build_report(simulator, theta, theta_true, states, failure_mask)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "lotka_volterra_prior_predictive_report.txt"
    report_path.write_text(report, encoding="utf-8")
    plot_prior_marginals(
        theta=theta,
        theta_true=theta_true,
        simulator=simulator,
        output_path=args.output_dir / "lotka_volterra_prior_marginals.png",
    )
    plot_prior_predictive(
        simulator=simulator,
        states=states,
        failure_mask=failure_mask,
        rng=rng,
        num_plot_trajectories=args.num_plot_trajectories,
        output_path=args.output_dir / "lotka_volterra_prior_predictive.png",
    )

    print(report)
    print(f"Saved: {report_path}")


def load_reference_simulator(reference_path: Path) -> tuple[LotkaVolterraSimulator, np.ndarray]:
    with h5py.File(reference_path, "r") as h5:
        simulator_metadata = json.loads(h5.attrs["simulator"])
        theta_true = h5["observations/theta_true"][...]

    simulator = LotkaVolterraSimulator(
        num_timepoints=int(simulator_metadata["num_timepoints"]),
        days=float(simulator_metadata["days"]),
        observation_noise_scale=float(simulator_metadata["observation_noise_scale"]),
        initial_state=tuple(simulator_metadata["initial_state"]),
        prior_log_mean=np.asarray(simulator_metadata["prior_log_mean"], dtype=float),
        prior_log_std=np.asarray(simulator_metadata["prior_log_std"], dtype=float),
        max_state=float(simulator_metadata["max_state"]),
        ode_rtol=float(simulator_metadata["ode_rtol"]),
        ode_atol=float(simulator_metadata["ode_atol"]),
    )
    return simulator, theta_true


def solve_prior_trajectories(
    simulator: LotkaVolterraSimulator,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.full(
        (theta.shape[0], simulator.num_timepoints, 2),
        np.nan,
        dtype=float,
    )
    failure_mask = np.zeros(theta.shape[0], dtype=bool)
    for index, theta_i in enumerate(theta):
        try:
            states[index] = simulator._solve_states(theta_i).T
        except Exception:
            failure_mask[index] = True
    return states, failure_mask


def build_report(
    simulator: LotkaVolterraSimulator,
    theta: np.ndarray,
    theta_true: np.ndarray,
    states: np.ndarray,
    failure_mask: np.ndarray,
) -> str:
    valid_states = states[~failure_mask]
    theta_names = ("alpha", "beta", "gamma", "delta")
    log_theta = np.log(theta)
    z_true = (np.log(theta_true) - simulator.prior_log_mean) / simulator.prior_log_std
    near_zero = valid_states < 1e-3
    near_max = valid_states > 0.95 * simulator.max_state
    valid_fraction = 1.0 - float(np.mean(failure_mask))

    lines = [
        "Lotka-Volterra prior predictive diagnostics",
        "",
        "Simulator:",
        f"  num_timepoints: {simulator.num_timepoints}",
        f"  days: {simulator.days}",
        f"  observation_layout: interleaved_prey_predator",
        f"  observation_noise: lognormal(scale={simulator.observation_noise_scale})",
        f"  initial_state: {simulator.initial_state.tolist()}",
        f"  prior_log_mean: {simulator.prior_log_mean.tolist()}",
        f"  prior_log_std: {simulator.prior_log_std.tolist()}",
        "",
        "Prior marginal quantiles:",
    ]
    for dim, name in enumerate(theta_names):
        q = np.quantile(theta[:, dim], [0.005, 0.05, 0.5, 0.95, 0.995])
        log_q = np.quantile(log_theta[:, dim], [0.005, 0.05, 0.5, 0.95, 0.995])
        lines.append(
            f"  {name}: q0.5={q[0]:.6g}, q5={q[1]:.6g}, "
            f"median={q[2]:.6g}, q95={q[3]:.6g}, q99.5={q[4]:.6g}; "
            f"log q5/50/95=({log_q[1]:.6g}, {log_q[2]:.6g}, {log_q[3]:.6g})"
        )

    lines.extend(
        [
            "",
            "Prior predictive pathology rates:",
            f"  ODE failure rate: {np.mean(failure_mask):.6g}",
            f"  valid trajectory fraction: {valid_fraction:.6g}",
            f"  any state near zero (<1e-3): {np.mean(np.any(near_zero, axis=(1, 2))):.6g}",
            f"  any state near max_state (>0.95 max): {np.mean(np.any(near_max, axis=(1, 2))):.6g}",
        ]
    )

    if valid_states.size > 0:
        prey = valid_states[:, :, 0]
        predator = valid_states[:, :, 1]
        for name, population in (("prey", prey), ("predator", predator)):
            q = np.quantile(population, [0.005, 0.05, 0.5, 0.95, 0.995])
            max_q = np.quantile(np.max(population, axis=1), [0.5, 0.95, 0.995])
            min_q = np.quantile(np.min(population, axis=1), [0.005, 0.05, 0.5])
            lines.extend(
                [
                    "",
                    f"{name} state distribution over all valid timepoints:",
                    f"  q0.5={q[0]:.6g}, q5={q[1]:.6g}, median={q[2]:.6g}, "
                    f"q95={q[3]:.6g}, q99.5={q[4]:.6g}",
                    f"  trajectory max q50/q95/q99.5=({max_q[0]:.6g}, {max_q[1]:.6g}, {max_q[2]:.6g})",
                    f"  trajectory min q0.5/q5/q50=({min_q[0]:.6g}, {min_q[1]:.6g}, {min_q[2]:.6g})",
                ]
            )

    lines.extend(["", "Reference theta_true prior z-scores:"])
    for obs_idx, z in enumerate(z_true):
        max_abs_z = float(np.max(np.abs(z)))
        lines.append(
            f"  obs {obs_idx:02d}: max_abs_z={max_abs_z:.3f}, "
            f"z={np.array2string(z, precision=3, separator=', ')}"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "  This prior is acceptable if failures/clipping are rare and the prior predictive",
            "  spans diverse but nondegenerate predator-prey dynamics. High near-zero or",
            "  near-max rates would indicate a prior that is numerically or scientifically weak.",
        ]
    )
    return "\n".join(lines) + "\n"


def plot_prior_marginals(
    *,
    theta: np.ndarray,
    theta_true: np.ndarray,
    simulator: LotkaVolterraSimulator,
    output_path: Path,
) -> None:
    names = ("alpha", "beta", "gamma", "delta")
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 5.5), squeeze=False)
    for dim, name in enumerate(names):
        axes[0, dim].hist(theta[:, dim], bins=60, color="C0", alpha=0.75, density=True)
        for value in theta_true[:, dim]:
            axes[0, dim].axvline(value, color="C3", alpha=0.45, linewidth=0.8)
        axes[0, dim].set_title(name)
        axes[0, dim].set_xlabel("theta")

        log_theta = np.log(theta[:, dim])
        axes[1, dim].hist(log_theta, bins=60, color="C1", alpha=0.75, density=True)
        for value in np.log(theta_true[:, dim]):
            axes[1, dim].axvline(value, color="C3", alpha=0.45, linewidth=0.8)
        axes[1, dim].axvline(simulator.prior_log_mean[dim], color="black", linewidth=1.1)
        axes[1, dim].set_xlabel("log(theta)")
    fig.suptitle("Lotka-Volterra prior marginals with reference truths")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_prior_predictive(
    *,
    simulator: LotkaVolterraSimulator,
    states: np.ndarray,
    failure_mask: np.ndarray,
    rng: np.random.Generator,
    num_plot_trajectories: int,
    output_path: Path,
) -> None:
    valid_states = states[~failure_mask]
    if valid_states.size == 0:
        return
    q05, q50, q95 = np.quantile(valid_states, [0.05, 0.5, 0.95], axis=0)
    num_lines = min(num_plot_trajectories, valid_states.shape[0])
    indices = rng.choice(valid_states.shape[0], size=num_lines, replace=False)
    time = simulator.timepoints

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), sharex=True)
    for population, (ax, name, color) in enumerate(
        zip(axes, ("prey", "predator"), ("C0", "C2"), strict=True)
    ):
        for index in indices:
            ax.plot(
                time,
                valid_states[index, :, population],
                color=color,
                alpha=0.12,
                linewidth=0.7,
            )
        ax.fill_between(
            time,
            q05[:, population],
            q95[:, population],
            color=color,
            alpha=0.22,
            label="90% prior band",
        )
        ax.plot(time, q50[:, population], color=color, linewidth=1.8, label="median")
        ax.set_yscale("log")
        ax.set_xlabel("time")
        ax.set_ylabel(name)
        ax.legend(fontsize="small")
    fig.suptitle("Lotka-Volterra prior predictive trajectories")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
