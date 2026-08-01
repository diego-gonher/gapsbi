from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gapsbi.references import load_reference_posteriors_hdf5


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a compact diagnostics report for a reference posterior artifact."
    )
    parser.add_argument("reference_path", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = build_report(args.reference_path)
    if args.output is None:
        output_path = (
            args.reference_path.parent
            / "diagnostics"
            / f"{args.reference_path.stem}_diagnostics.txt"
        )
    else:
        output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved: {output_path}")


def build_report(reference_path: Path) -> str:
    observations, theta_samples, metadata, diagnostics = load_reference_posteriors_hdf5(
        reference_path
    )
    lines = [
        f"Reference artifact: {reference_path}",
        f"Problem: {metadata.get('problem')}",
        f"Posterior method: {metadata.get('posterior_method')}",
        f"Reference version: {metadata.get('reference_version')}",
        f"theta_true shape: {observations['theta_true'].shape}",
        f"x_full shape: {observations['x_full'].shape}",
        f"theta_samples shape: {theta_samples.shape}",
        f"theta_scaled: {metadata.get('theta_scaled')}",
        f"x_scaled: {metadata.get('x_scaled')}",
    ]
    if "num_prior_bound_violations" in diagnostics:
        lines.append(
            f"Prior-bound violations: {np.asarray(diagnostics['num_prior_bound_violations']).item()}"
        )
    if "grid_ess" in diagnostics:
        lines.extend(
            [
                "",
                "Grid diagnostics:",
                _summary_line("grid_ess", diagnostics["grid_ess"]),
                _summary_line("grid_boundary_mass", diagnostics["grid_boundary_mass"]),
            ]
        )
    if "validation_mean_l2_delta_vs_final_grid" in diagnostics:
        lines.extend(
            [
                _summary_line(
                    "validation_mean_l2_delta_vs_final_grid",
                    diagnostics["validation_mean_l2_delta_vs_final_grid"],
                ),
                _summary_line(
                    "validation_cov_fro_delta_vs_final_grid",
                    diagnostics["validation_cov_fro_delta_vs_final_grid"],
                ),
            ]
        )
    if "spotcheck_mean_l2_delta_vs_final_grid" in diagnostics:
        lines.extend(
            [
                _summary_line(
                    "spotcheck_mean_l2_delta_vs_final_grid",
                    diagnostics["spotcheck_mean_l2_delta_vs_final_grid"],
                ),
                _summary_line(
                    "spotcheck_cov_fro_delta_vs_final_grid",
                    diagnostics["spotcheck_cov_fro_delta_vs_final_grid"],
                ),
            ]
        )
    if "acceptance_fraction" in diagnostics:
        lines.extend(
            [
                "",
                "MCMC diagnostics:",
                f"num_walkers: {metadata.get('num_walkers')}",
                f"burn_in_steps: {metadata.get('burn_in_steps')}",
                f"production_steps: {metadata.get('production_steps')}",
                _summary_line("acceptance_fraction", diagnostics["acceptance_fraction"]),
                _summary_line("split_rhat", diagnostics["split_rhat"]),
                _summary_line("emcee_ess", diagnostics["emcee_ess"]),
                _summary_line("autocorr_time", diagnostics["autocorr_time"]),
                _summary_line(
                    "walker_group_mean_l2_max",
                    diagnostics["walker_group_mean_l2_max"],
                ),
                _summary_line(
                    "walker_group_std_l2_max",
                    diagnostics["walker_group_std_l2_max"],
                ),
                _summary_line(
                    "validation_mean_l2_delta",
                    diagnostics["validation_mean_l2_delta"],
                ),
                _summary_line(
                    "validation_cov_fro_delta",
                    diagnostics["validation_cov_fro_delta"],
                ),
            ]
        )
    if "log_theta_split_rhat" in diagnostics:
        lines.extend(
            [
                _summary_line("log_theta_split_rhat", diagnostics["log_theta_split_rhat"]),
                _summary_line("log_theta_emcee_ess", diagnostics["log_theta_emcee_ess"]),
            ]
        )
    if "ode_failure_count" in diagnostics:
        lines.extend(
            [
                _summary_line("ode_failure_count", diagnostics["ode_failure_count"]),
                _summary_line(
                    "posterior_predictive_log_rmse_mean",
                    diagnostics["posterior_predictive_log_rmse_mean"],
                ),
                _summary_line(
                    "posterior_predictive_log_rmse_q05",
                    diagnostics["posterior_predictive_log_rmse_q05"],
                ),
                _summary_line(
                    "posterior_predictive_log_rmse_q95",
                    diagnostics["posterior_predictive_log_rmse_q95"],
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def _summary_line(name: str, values: np.ndarray) -> str:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return f"{name}: no finite values"
    return (
        f"{name}: min={np.min(finite):.6g}, "
        f"median={np.median(finite):.6g}, max={np.max(finite):.6g}"
    )


if __name__ == "__main__":
    main()
