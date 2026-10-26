from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_ndtr, logsumexp
from scipy.stats import chi2, norm, truncnorm

from gapsbi.masks import MeanNormalizedSelfCensoringMNARMask, PointMCARMask, LotkaVolterraLogTotalMNARMask
from gapsbi.simulators import GLMSimulator, GLUSimulator, LotkaVolterraSimulator, OUPSimulator


REFERENCE_SCHEMA_NAME = "gapsbi_reference_posteriors"
REFERENCE_SCHEMA_VERSION = 1
DEFAULT_REFERENCE_VERSION = "reference_posteriors_v1"
PARTIAL_DATA_REFERENCE_SCHEMA_NAME = "gapsbi_partial_data_reference_posteriors"
PARTIAL_DATA_REFERENCE_SCHEMA_VERSION = 1
DEFAULT_PARTIAL_DATA_REFERENCE_VERSION = "partial_data_references_v1"
REFERENCE_PROBLEMS = ("glu", "oup", "glm", "lotka_volterra")


def default_full_reference_path(
    problem: str,
    *,
    root: str | os.PathLike[str] | None = None,
    reference_version: str = DEFAULT_REFERENCE_VERSION,
) -> Path:
    """Return the current physical path for one full-data reference artifact.

    The versioned default remains deliberate provenance. Callers may supply a
    root for a future public-facing reference alias without changing artifact
    schemas or existing configuration paths.
    """
    if problem not in REFERENCE_PROBLEMS:
        raise ValueError(f"Unknown reference problem: {problem!r}.")
    base = Path(root) if root is not None else Path("references") / reference_version
    return base / f"{problem}_references.h5"


def default_partial_reference_path(
    problem: str,
    mechanism: str,
    missing_fraction: float,
    *,
    root: str | os.PathLike[str] | None = None,
    reference_version: str = DEFAULT_PARTIAL_DATA_REFERENCE_VERSION,
) -> Path:
    """Return the current physical path for a partial-data reference artifact.

    LV/MNAR at 50% is intentionally sparse and retains its validated
    ``successful_references`` filename.
    """
    if problem not in REFERENCE_PROBLEMS:
        raise ValueError(f"Unknown reference problem: {problem!r}.")
    if mechanism not in {"mcar", "mnar"}:
        raise ValueError(f"Unknown partial-reference mechanism: {mechanism!r}.")
    if not 0.0 <= missing_fraction <= 1.0:
        raise ValueError("missing_fraction must be in [0, 1].")
    tag = f"eps{round(100 * missing_fraction):03d}"
    suffix = (
        "_successful_references.h5"
        if (problem, mechanism, missing_fraction) == ("lotka_volterra", "mnar", 0.50)
        else "_references.h5"
    )
    base = Path(root) if root is not None else Path("references") / reference_version
    return base / f"{problem}_{mechanism}_{tag}{suffix}"


def partial_reference_observations(
    full_observations: dict[str, np.ndarray],
    mask: np.ndarray,
    *,
    reference_indices: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build the shared partial-reference observation contract.

    This preserves the existing ``x_obs = x_full * mask`` convention and is a
    guard against a generator writing observations that do not correspond to
    its immutable full-data source.
    """
    if "theta_true" not in full_observations or "x_full" not in full_observations:
        raise ValueError("full_observations must contain theta_true and x_full.")
    theta_true = np.asarray(full_observations["theta_true"])
    x_full = np.asarray(full_observations["x_full"])
    mask_array = np.asarray(mask)
    if theta_true.ndim != 2 or x_full.ndim != 2 or mask_array.shape != x_full.shape:
        raise ValueError("theta_true, x_full, and mask have incompatible shapes.")
    observations = {
        "theta_true": theta_true,
        "x_full": x_full,
        "x_obs": x_full * mask_array,
        "mask": mask_array,
    }
    if reference_indices is not None:
        indices = np.asarray(reference_indices, dtype=np.int64)
        if indices.shape != (theta_true.shape[0],):
            raise ValueError("reference_indices must have one entry per observation.")
        observations["reference_indices"] = indices
    return observations


def generate_glu_mcar_partial_data_reference_posteriors(
    *,
    num_observations: int = 10,
    num_reference_samples: int = 10_000,
    observation_seed: int = 12_345,
    posterior_seed: int = 23_456,
    dim: int = 10,
    prior_bound: float = 1.0,
    simulator_scale: float = 0.1,
    missing_fraction: float,
    reference_mask_seed: int = 987_654,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, np.ndarray]]:
    """Generate GLU MCAR references for ``p(theta | x_obs, mask)``.

    The mask draw deliberately matches the current evaluation code: one batch
    call to ``PointMCARMask.generate`` with ``default_rng(reference_mask_seed)``.
    """
    simulator = GLUSimulator(
        dim=dim,
        prior_bound=prior_bound,
        simulator_scale=simulator_scale,
    )
    observation_rng = np.random.default_rng(observation_seed)
    posterior_rng = np.random.default_rng(posterior_seed)
    mask_generator = PointMCARMask(missing_fraction=missing_fraction)

    theta_true = simulator.sample_theta(num_observations, observation_rng)
    x_full = simulator.simulate(theta_true, observation_rng)
    mask = mask_generator.generate(
        x_full=x_full,
        theta=theta_true,
        rng=np.random.default_rng(reference_mask_seed),
    )
    x_obs = x_full * mask
    theta_samples = sample_glu_mcar_partial_data_reference_posterior(
        x_obs=x_obs,
        mask=mask,
        prior_low=simulator.prior_low,
        prior_high=simulator.prior_high,
        simulator_scale=simulator.simulator_scale,
        num_reference_samples=num_reference_samples,
        rng=posterior_rng,
    )
    observations = {
        "theta_true": theta_true,
        "x_full": x_full,
        "x_obs": x_obs,
        "mask": mask,
    }
    metadata = {
        "problem": "glu",
        "posterior_method": "analytic_mcar_partial_truncated_gaussian_uniform",
        "posterior_conditioning": "theta_given_x_obs_mask",
        "missingness_mechanism": "mcar",
        "mask": mask_generator.metadata(),
        "reference_mask_seed": int(reference_mask_seed),
        "full_data_reference_version": DEFAULT_REFERENCE_VERSION,
        "observation_seed": int(observation_seed),
        "posterior_seed": int(posterior_seed),
        "simulator": simulator.metadata(),
        "theta_scaled": False,
        "x_scaled": False,
    }
    diagnostics = {
        "posterior_sample_mean": theta_samples.mean(axis=1),
        "posterior_sample_std": theta_samples.std(axis=1),
        "prior_low": simulator.prior_low,
        "prior_high": simulator.prior_high,
        "num_prior_bound_violations": np.array(
            np.count_nonzero(
                (theta_samples < simulator.prior_low)
                | (theta_samples > simulator.prior_high)
            ),
            dtype=np.int64,
        ),
    }
    return observations, metadata, {"theta_samples": theta_samples, **diagnostics}


def generate_glu_reference_posteriors(
    *,
    num_observations: int = 10,
    num_reference_samples: int = 10_000,
    observation_seed: int = 12_345,
    posterior_seed: int = 23_456,
    dim: int = 10,
    prior_bound: float = 1.0,
    simulator_scale: float = 0.1,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, np.ndarray]]:
    """Generate fixed GLU observations and analytic truncated-Gaussian references."""
    if num_observations <= 0:
        raise ValueError("num_observations must be positive.")
    if num_reference_samples <= 0:
        raise ValueError("num_reference_samples must be positive.")
    if simulator_scale <= 0:
        raise ValueError("simulator_scale must be positive for a continuous posterior.")

    simulator = GLUSimulator(
        dim=dim,
        prior_bound=prior_bound,
        simulator_scale=simulator_scale,
    )
    observation_rng = np.random.default_rng(observation_seed)
    posterior_rng = np.random.default_rng(posterior_seed)

    theta_true = simulator.sample_theta(num_observations, observation_rng)
    x_full = simulator.simulate(theta_true, observation_rng)
    theta_samples = sample_glu_reference_posterior(
        x_full=x_full,
        prior_low=simulator.prior_low,
        prior_high=simulator.prior_high,
        simulator_scale=simulator.simulator_scale,
        num_reference_samples=num_reference_samples,
        rng=posterior_rng,
    )
    observations = {
        "theta_true": theta_true,
        "x_full": x_full,
    }
    metadata = {
        "problem": "glu",
        "posterior_method": "analytic_truncated_gaussian",
        "observation_seed": int(observation_seed),
        "posterior_seed": int(posterior_seed),
        "simulator": simulator.metadata(),
        "theta_scaled": False,
        "x_scaled": False,
    }
    diagnostics = {
        "posterior_sample_mean": theta_samples.mean(axis=1),
        "posterior_sample_std": theta_samples.std(axis=1),
        "prior_low": simulator.prior_low,
        "prior_high": simulator.prior_high,
        "num_prior_bound_violations": np.array(
            np.count_nonzero(
                (theta_samples < simulator.prior_low)
                | (theta_samples > simulator.prior_high)
            ),
            dtype=np.int64,
        ),
    }
    return observations, metadata, {"theta_samples": theta_samples, **diagnostics}


def generate_oup_reference_posteriors(
    *,
    num_observations: int = 10,
    num_reference_samples: int = 10_000,
    observation_seed: int = 12_345,
    posterior_seed: int = 23_456,
    grid_resolution: int = 800,
    validation_grid_resolutions: tuple[int, ...] = (400,),
    spotcheck_grid_resolution: int | None = 1200,
    spotcheck_observation_indices: tuple[int, ...] = (0, 1),
    progress: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, np.ndarray]]:
    """Generate fixed OUP observations and grid-based reference posteriors."""
    if num_observations <= 0:
        raise ValueError("num_observations must be positive.")
    if num_reference_samples <= 0:
        raise ValueError("num_reference_samples must be positive.")
    if grid_resolution <= 1:
        raise ValueError("grid_resolution must be greater than 1.")

    simulator = OUPSimulator()
    observation_rng = np.random.default_rng(observation_seed)
    posterior_rng = np.random.default_rng(posterior_seed)

    theta_true = simulator.sample_theta(num_observations, observation_rng)
    x_full = simulator.simulate(theta_true, observation_rng)

    theta_samples = np.empty(
        (num_observations, num_reference_samples, simulator.theta_dim),
        dtype=float,
    )
    grid_mean = np.empty((num_observations, simulator.theta_dim), dtype=float)
    grid_cov = np.empty((num_observations, simulator.theta_dim, simulator.theta_dim), dtype=float)
    grid_ess = np.empty(num_observations, dtype=float)
    grid_boundary_mass = np.empty(num_observations, dtype=float)
    grid_log_normalizer = np.empty(num_observations, dtype=float)

    final_posteriors = []
    for obs_idx in range(num_observations):
        if progress:
            print(
                f"OUP reference {obs_idx + 1}/{num_observations}: "
                f"computing {grid_resolution} x {grid_resolution} grid",
                flush=True,
            )
        posterior = compute_oup_grid_posterior(
            x_full[obs_idx],
            simulator=simulator,
            grid_resolution=grid_resolution,
        )
        final_posteriors.append(posterior)
        theta_samples[obs_idx] = sample_oup_grid_posterior(
            posterior,
            num_reference_samples=num_reference_samples,
            rng=posterior_rng,
        )
        grid_mean[obs_idx] = posterior["mean"]
        grid_cov[obs_idx] = posterior["cov"]
        grid_ess[obs_idx] = posterior["ess"]
        grid_boundary_mass[obs_idx] = posterior["boundary_mass"]
        grid_log_normalizer[obs_idx] = posterior["log_normalizer"]
        if progress:
            print(
                f"OUP reference {obs_idx + 1}/{num_observations}: done "
                f"(boundary_mass={grid_boundary_mass[obs_idx]:.6g}, "
                f"ess={grid_ess[obs_idx]:.6g})",
                flush=True,
            )

    diagnostics: dict[str, np.ndarray] = {
        "posterior_sample_mean": theta_samples.mean(axis=1),
        "posterior_sample_std": theta_samples.std(axis=1),
        "grid_mean": grid_mean,
        "grid_cov": grid_cov,
        "grid_ess": grid_ess,
        "grid_boundary_mass": grid_boundary_mass,
        "grid_log_normalizer": grid_log_normalizer,
        "prior_low": simulator.prior_low,
        "prior_high": simulator.prior_high,
        "num_prior_bound_violations": np.array(
            np.count_nonzero(
                (theta_samples < simulator.prior_low)
                | (theta_samples > simulator.prior_high)
            ),
            dtype=np.int64,
        ),
    }
    if validation_grid_resolutions:
        if progress:
            resolutions = ", ".join(str(value) for value in validation_grid_resolutions)
            print(f"OUP validation grids: {resolutions}", flush=True)
        diagnostics.update(
            compute_oup_grid_resolution_diagnostics(
                x_full,
                simulator=simulator,
                reference_posteriors=final_posteriors,
                validation_grid_resolutions=validation_grid_resolutions,
            )
        )
    if spotcheck_grid_resolution is not None:
        if progress:
            indices = ", ".join(str(value) for value in spotcheck_observation_indices)
            print(
                f"OUP spotcheck grid {spotcheck_grid_resolution} for observations: {indices}",
                flush=True,
            )
        diagnostics.update(
            compute_oup_grid_resolution_diagnostics(
                x_full,
                simulator=simulator,
                reference_posteriors=final_posteriors,
                validation_grid_resolutions=(spotcheck_grid_resolution,),
                observation_indices=spotcheck_observation_indices,
                prefix="spotcheck",
            )
        )

    observations = {
        "theta_true": theta_true,
        "x_full": x_full,
    }
    metadata = {
        "problem": "oup",
        "posterior_method": "deterministic_grid",
        "observation_seed": int(observation_seed),
        "posterior_seed": int(posterior_seed),
        "simulator": simulator.metadata(),
        "theta_scaled": False,
        "x_scaled": False,
        "grid_resolution": int(grid_resolution),
        "validation_grid_resolutions": list(validation_grid_resolutions),
        "spotcheck_grid_resolution": (
            None if spotcheck_grid_resolution is None else int(spotcheck_grid_resolution)
        ),
        "spotcheck_observation_indices": list(spotcheck_observation_indices),
    }
    return observations, metadata, {"theta_samples": theta_samples, **diagnostics}


def generate_glm_reference_posteriors(
    *,
    num_observations: int = 10,
    num_reference_samples: int = 10_000,
    observation_seed: int = 12_345,
    posterior_seed: int = 23_456,
    dim: int = 10,
    prior_bound: float | None = None,
    duration: int = 100,
    stimulus_seed: int = 42,
    num_walkers: int = 128,
    burn_in_steps: int = 3_000,
    production_steps: int = 3_000,
    initial_scale: float = 0.05,
    trace_num_steps: int = 1_000,
    trace_num_walkers: int = 8,
    validation_num_ensembles: int = 2,
    validation_observation_indices: tuple[int, ...] = (0, 1),
    progress: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, np.ndarray]]:
    """Generate fixed raw-GLM observations and emcee reference posteriors."""
    if num_observations <= 0:
        raise ValueError("num_observations must be positive.")
    if num_reference_samples <= 0:
        raise ValueError("num_reference_samples must be positive.")
    if num_walkers < 2 * dim:
        raise ValueError("num_walkers must be at least 2 * dim for emcee.")
    if burn_in_steps <= 0 or production_steps <= 0:
        raise ValueError("burn_in_steps and production_steps must be positive.")

    simulator = GLMSimulator(
        dim=dim,
        prior_bound=prior_bound,
        duration=duration,
        stimulus_seed=stimulus_seed,
        summary="raw",
    )
    observation_rng = np.random.default_rng(observation_seed)
    posterior_rng = np.random.default_rng(posterior_seed)

    theta_true = simulator.sample_theta(num_observations, observation_rng)
    x_full = simulator.simulate(theta_true, observation_rng)
    theta_samples = np.empty(
        (num_observations, num_reference_samples, simulator.theta_dim),
        dtype=float,
    )

    acceptance_fraction = np.empty(num_observations, dtype=float)
    autocorr_time = np.full((num_observations, simulator.theta_dim), np.nan, dtype=float)
    emcee_ess = np.full((num_observations, simulator.theta_dim), np.nan, dtype=float)
    split_rhat = np.full((num_observations, simulator.theta_dim), np.nan, dtype=float)
    walker_group_mean_l2_max = np.empty(num_observations, dtype=float)
    walker_group_std_l2_max = np.empty(num_observations, dtype=float)
    map_estimate = np.empty((num_observations, simulator.theta_dim), dtype=float)
    map_log_prob = np.empty(num_observations, dtype=float)
    posterior_sample_mean = np.empty((num_observations, simulator.theta_dim), dtype=float)
    posterior_sample_std = np.empty((num_observations, simulator.theta_dim), dtype=float)
    trace_steps = min(trace_num_steps, production_steps)
    trace_walkers = min(trace_num_walkers, num_walkers)
    mcmc_trace = np.empty(
        (num_observations, trace_steps, trace_walkers, simulator.theta_dim),
        dtype=float,
    )
    validation_mean_l2_delta = np.full(
        (validation_num_ensembles, len(validation_observation_indices)),
        np.nan,
        dtype=float,
    )
    validation_cov_fro_delta = np.full_like(validation_mean_l2_delta, np.nan)

    for obs_idx in range(num_observations):
        if progress:
            print(
                f"GLM reference {obs_idx + 1}/{num_observations}: running main ensemble",
                flush=True,
            )
        result = run_glm_emcee_reference(
            x_full[obs_idx],
            simulator=simulator,
            num_walkers=num_walkers,
            burn_in_steps=burn_in_steps,
            production_steps=production_steps,
            initial_scale=initial_scale,
            rng=posterior_rng,
        )
        samples_flat = result["chain"].reshape(-1, simulator.theta_dim)
        sample_indices = posterior_rng.choice(
            samples_flat.shape[0],
            size=num_reference_samples,
            replace=samples_flat.shape[0] < num_reference_samples,
        )
        theta_samples[obs_idx] = samples_flat[sample_indices]
        acceptance_fraction[obs_idx] = float(result["acceptance_fraction"])
        autocorr_time[obs_idx] = np.asarray(result["autocorr_time"], dtype=float)
        emcee_ess[obs_idx] = np.asarray(result["emcee_ess"], dtype=float)
        split_rhat[obs_idx] = np.asarray(result["split_rhat"], dtype=float)
        walker_group_mean_l2_max[obs_idx] = float(result["walker_group_mean_l2_max"])
        walker_group_std_l2_max[obs_idx] = float(result["walker_group_std_l2_max"])
        map_estimate[obs_idx] = np.asarray(result["map_estimate"], dtype=float)
        map_log_prob[obs_idx] = float(result["map_log_prob"])
        posterior_sample_mean[obs_idx] = theta_samples[obs_idx].mean(axis=0)
        posterior_sample_std[obs_idx] = theta_samples[obs_idx].std(axis=0)
        trace_idx = np.linspace(0, production_steps - 1, trace_steps, dtype=int)
        mcmc_trace[obs_idx] = result["chain"][trace_idx, :trace_walkers, :]
        if progress:
            print(
                f"GLM reference {obs_idx + 1}/{num_observations}: done "
                f"(acceptance={acceptance_fraction[obs_idx]:.3f}, "
                f"max_rhat={np.nanmax(split_rhat[obs_idx]):.3f})",
                flush=True,
            )

    for local_idx, obs_idx in enumerate(validation_observation_indices):
        if obs_idx < 0 or obs_idx >= num_observations:
            continue
        reference_mean = posterior_sample_mean[obs_idx]
        reference_cov = np.cov(theta_samples[obs_idx], rowvar=False)
        for ensemble_idx in range(validation_num_ensembles):
            if progress:
                print(
                    "GLM validation "
                    f"{ensemble_idx + 1}/{validation_num_ensembles} "
                    f"for observation {obs_idx}",
                    flush=True,
                )
            result = run_glm_emcee_reference(
                x_full[obs_idx],
                simulator=simulator,
                num_walkers=num_walkers,
                burn_in_steps=burn_in_steps,
                production_steps=production_steps,
                initial_scale=initial_scale,
                rng=posterior_rng,
            )
            samples_flat = result["chain"].reshape(-1, simulator.theta_dim)
            validation_mean_l2_delta[ensemble_idx, local_idx] = float(
                np.linalg.norm(samples_flat.mean(axis=0) - reference_mean)
            )
            validation_cov_fro_delta[ensemble_idx, local_idx] = float(
                np.linalg.norm(np.cov(samples_flat, rowvar=False) - reference_cov, ord="fro")
            )

    observations = {
        "theta_true": theta_true,
        "x_full": x_full,
    }
    metadata = {
        "problem": "glm",
        "posterior_method": "emcee_ensemble_mcmc",
        "observation_seed": int(observation_seed),
        "posterior_seed": int(posterior_seed),
        "simulator": simulator.metadata(),
        "theta_scaled": False,
        "x_scaled": False,
        "num_walkers": int(num_walkers),
        "burn_in_steps": int(burn_in_steps),
        "production_steps": int(production_steps),
        "initial_scale": float(initial_scale),
        "validation_num_ensembles": int(validation_num_ensembles),
        "validation_observation_indices": list(validation_observation_indices),
    }
    diagnostics = {
        "posterior_sample_mean": posterior_sample_mean,
        "posterior_sample_std": posterior_sample_std,
        "prior_mean": simulator.prior_mean,
        "prior_covariance": simulator.prior_covariance,
        "prior_precision": simulator.prior_precision,
        "num_nonfinite_prior_log_prob": np.array(
            np.count_nonzero(~np.isfinite(simulator.log_prior(theta_samples.reshape(-1, simulator.theta_dim)))),
            dtype=np.int64,
        ),
        "acceptance_fraction": acceptance_fraction,
        "autocorr_time": autocorr_time,
        "emcee_ess": emcee_ess,
        "split_rhat": split_rhat,
        "walker_group_mean_l2_max": walker_group_mean_l2_max,
        "walker_group_std_l2_max": walker_group_std_l2_max,
        "map_estimate": map_estimate,
        "map_log_prob": map_log_prob,
        "mcmc_trace": mcmc_trace,
        "validation_observation_indices": np.asarray(validation_observation_indices, dtype=np.int64),
        "validation_mean_l2_delta": validation_mean_l2_delta,
        "validation_cov_fro_delta": validation_cov_fro_delta,
    }
    return observations, metadata, {"theta_samples": theta_samples, **diagnostics}


def generate_lotka_volterra_reference_posteriors(
    *,
    num_observations: int = 10,
    num_reference_samples: int = 10_000,
    observation_seed: int = 12_345,
    posterior_seed: int = 23_456,
    num_timepoints: int = 50,
    days: float = 20.0,
    observation_noise_scale: float = 0.1,
    num_walkers: int = 64,
    burn_in_steps: int = 1_500,
    production_steps: int = 3_000,
    initial_scale: float = 0.08,
    trace_num_steps: int = 1_000,
    trace_num_walkers: int = 8,
    validation_num_ensembles: int = 1,
    validation_observation_indices: tuple[int, ...] = (0, 1),
    progress: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, np.ndarray]]:
    """Generate fixed Lotka-Volterra observations and log-space emcee references."""
    if num_observations <= 0:
        raise ValueError("num_observations must be positive.")
    if num_reference_samples <= 0:
        raise ValueError("num_reference_samples must be positive.")
    if num_walkers < 2 * 4:
        raise ValueError("num_walkers must be at least 2 * theta_dim for emcee.")
    if burn_in_steps <= 0 or production_steps <= 0:
        raise ValueError("burn_in_steps and production_steps must be positive.")

    simulator = LotkaVolterraSimulator(
        num_timepoints=num_timepoints,
        days=days,
        observation_noise_scale=observation_noise_scale,
    )
    observation_rng = np.random.default_rng(observation_seed)
    posterior_rng = np.random.default_rng(posterior_seed)

    theta_true = simulator.sample_theta(num_observations, observation_rng)
    x_full = simulator.simulate(theta_true, observation_rng)
    theta_samples = np.empty(
        (num_observations, num_reference_samples, simulator.theta_dim),
        dtype=float,
    )

    acceptance_fraction = np.empty(num_observations, dtype=float)
    autocorr_time = np.full((num_observations, simulator.theta_dim), np.nan, dtype=float)
    emcee_ess = np.full((num_observations, simulator.theta_dim), np.nan, dtype=float)
    split_rhat = np.full((num_observations, simulator.theta_dim), np.nan, dtype=float)
    log_theta_split_rhat = np.full_like(split_rhat, np.nan)
    log_theta_emcee_ess = np.full_like(emcee_ess, np.nan)
    walker_group_mean_l2_max = np.empty(num_observations, dtype=float)
    walker_group_std_l2_max = np.empty(num_observations, dtype=float)
    map_estimate = np.empty((num_observations, simulator.theta_dim), dtype=float)
    map_log_theta = np.empty_like(map_estimate)
    map_log_prob = np.empty(num_observations, dtype=float)
    posterior_sample_mean = np.empty((num_observations, simulator.theta_dim), dtype=float)
    posterior_sample_std = np.empty((num_observations, simulator.theta_dim), dtype=float)
    log_theta_sample_mean = np.empty_like(posterior_sample_mean)
    log_theta_sample_std = np.empty_like(posterior_sample_std)
    ode_failure_count = np.empty(num_observations, dtype=np.int64)
    posterior_predictive_log_rmse_mean = np.empty(num_observations, dtype=float)
    posterior_predictive_log_rmse_q05 = np.empty(num_observations, dtype=float)
    posterior_predictive_log_rmse_q95 = np.empty(num_observations, dtype=float)
    trace_steps = min(trace_num_steps, production_steps)
    trace_walkers = min(trace_num_walkers, num_walkers)
    mcmc_trace = np.empty(
        (num_observations, trace_steps, trace_walkers, simulator.theta_dim),
        dtype=float,
    )
    log_theta_mcmc_trace = np.empty_like(mcmc_trace)
    validation_mean_l2_delta = np.full(
        (validation_num_ensembles, len(validation_observation_indices)),
        np.nan,
        dtype=float,
    )
    validation_cov_fro_delta = np.full_like(validation_mean_l2_delta, np.nan)

    for obs_idx in range(num_observations):
        if progress:
            print(
                f"LV reference {obs_idx + 1}/{num_observations}: running MCMC",
                flush=True,
            )
        result = run_lotka_volterra_emcee_reference(
            x_full[obs_idx],
            simulator=simulator,
            num_walkers=num_walkers,
            burn_in_steps=burn_in_steps,
            production_steps=production_steps,
            initial_scale=initial_scale,
            initial_log_theta=np.log(theta_true[obs_idx]),
            rng=posterior_rng,
        )
        chain = np.asarray(result["chain"], dtype=float)
        log_chain = np.asarray(result["log_theta_chain"], dtype=float)
        samples_flat = chain.reshape(-1, simulator.theta_dim)
        sample_indices = posterior_rng.choice(
            samples_flat.shape[0],
            size=num_reference_samples,
            replace=samples_flat.shape[0] < num_reference_samples,
        )
        theta_samples[obs_idx] = samples_flat[sample_indices]
        acceptance_fraction[obs_idx] = float(result["acceptance_fraction"])
        autocorr_time[obs_idx] = np.asarray(result["autocorr_time"], dtype=float)
        emcee_ess[obs_idx] = np.asarray(result["emcee_ess"], dtype=float)
        split_rhat[obs_idx] = np.asarray(result["split_rhat"], dtype=float)
        log_theta_split_rhat[obs_idx] = np.asarray(result["log_theta_split_rhat"], dtype=float)
        log_theta_emcee_ess[obs_idx] = np.asarray(result["log_theta_emcee_ess"], dtype=float)
        walker_group_mean_l2_max[obs_idx] = float(result["walker_group_mean_l2_max"])
        walker_group_std_l2_max[obs_idx] = float(result["walker_group_std_l2_max"])
        map_estimate[obs_idx] = np.asarray(result["map_estimate"], dtype=float)
        map_log_theta[obs_idx] = np.asarray(result["map_log_theta"], dtype=float)
        map_log_prob[obs_idx] = float(result["map_log_prob"])
        posterior_sample_mean[obs_idx] = theta_samples[obs_idx].mean(axis=0)
        posterior_sample_std[obs_idx] = theta_samples[obs_idx].std(axis=0)
        log_theta_samples = np.log(theta_samples[obs_idx])
        log_theta_sample_mean[obs_idx] = log_theta_samples.mean(axis=0)
        log_theta_sample_std[obs_idx] = log_theta_samples.std(axis=0)
        ode_failure_count[obs_idx] = int(result["ode_failure_count"])
        rmse = lotka_volterra_posterior_predictive_log_rmse(
            theta_samples[obs_idx],
            x_full[obs_idx],
            simulator=simulator,
        )
        posterior_predictive_log_rmse_mean[obs_idx] = float(np.mean(rmse))
        posterior_predictive_log_rmse_q05[obs_idx] = float(np.quantile(rmse, 0.05))
        posterior_predictive_log_rmse_q95[obs_idx] = float(np.quantile(rmse, 0.95))
        trace_idx = np.linspace(0, production_steps - 1, trace_steps, dtype=int)
        mcmc_trace[obs_idx] = chain[trace_idx, :trace_walkers, :]
        log_theta_mcmc_trace[obs_idx] = log_chain[trace_idx, :trace_walkers, :]
        if progress:
            max_rhat = float(np.nanmax(log_theta_split_rhat[obs_idx]))
            min_ess = float(np.nanmin(log_theta_emcee_ess[obs_idx]))
            print(
                f"LV reference {obs_idx + 1}/{num_observations}: "
                f"accept={acceptance_fraction[obs_idx]:.3f}, "
                f"max_log_rhat={max_rhat:.3f}, min_log_ess={min_ess:.1f}",
                flush=True,
            )

    for local_idx, obs_idx in enumerate(validation_observation_indices):
        if obs_idx < 0 or obs_idx >= num_observations:
            continue
        reference_mean = posterior_sample_mean[obs_idx]
        reference_cov = np.cov(theta_samples[obs_idx], rowvar=False)
        for ensemble_idx in range(validation_num_ensembles):
            if progress:
                print(
                    "LV validation ensemble "
                    f"{ensemble_idx + 1}/{validation_num_ensembles} "
                    f"for observation {obs_idx}",
                    flush=True,
                )
            result = run_lotka_volterra_emcee_reference(
                x_full[obs_idx],
                simulator=simulator,
                num_walkers=num_walkers,
                burn_in_steps=burn_in_steps,
                production_steps=production_steps,
                initial_scale=initial_scale,
                rng=posterior_rng,
            )
            samples_flat = np.asarray(result["chain"], dtype=float).reshape(
                -1, simulator.theta_dim
            )
            validation_mean_l2_delta[ensemble_idx, local_idx] = float(
                np.linalg.norm(samples_flat.mean(axis=0) - reference_mean)
            )
            validation_cov_fro_delta[ensemble_idx, local_idx] = float(
                np.linalg.norm(np.cov(samples_flat, rowvar=False) - reference_cov, ord="fro")
            )

    observations = {
        "theta_true": theta_true,
        "x_full": x_full,
    }
    metadata = {
        "problem": "lotka_volterra",
        "posterior_method": "emcee_log_space_exact_likelihood",
        "observation_seed": int(observation_seed),
        "posterior_seed": int(posterior_seed),
        "simulator": simulator.metadata(),
        "theta_scaled": False,
        "x_scaled": False,
        "num_walkers": int(num_walkers),
        "burn_in_steps": int(burn_in_steps),
        "production_steps": int(production_steps),
        "initial_scale": float(initial_scale),
        "sampling_space": "log_theta",
        "map_initialization": "prior_starts_plus_theta_true",
        "validation_num_ensembles": int(validation_num_ensembles),
        "validation_observation_indices": list(validation_observation_indices),
    }
    diagnostics = {
        "posterior_sample_mean": posterior_sample_mean,
        "posterior_sample_std": posterior_sample_std,
        "log_theta_sample_mean": log_theta_sample_mean,
        "log_theta_sample_std": log_theta_sample_std,
        "prior_log_mean": simulator.prior_log_mean,
        "prior_log_std": simulator.prior_log_std,
        "num_nonpositive_theta_samples": np.array(
            np.count_nonzero(theta_samples <= 0),
            dtype=np.int64,
        ),
        "acceptance_fraction": acceptance_fraction,
        "autocorr_time": autocorr_time,
        "emcee_ess": emcee_ess,
        "split_rhat": split_rhat,
        "log_theta_split_rhat": log_theta_split_rhat,
        "log_theta_emcee_ess": log_theta_emcee_ess,
        "walker_group_mean_l2_max": walker_group_mean_l2_max,
        "walker_group_std_l2_max": walker_group_std_l2_max,
        "map_estimate": map_estimate,
        "map_log_theta": map_log_theta,
        "map_log_prob": map_log_prob,
        "mcmc_trace": mcmc_trace,
        "log_theta_mcmc_trace": log_theta_mcmc_trace,
        "ode_failure_count": ode_failure_count,
        "posterior_predictive_log_rmse_mean": posterior_predictive_log_rmse_mean,
        "posterior_predictive_log_rmse_q05": posterior_predictive_log_rmse_q05,
        "posterior_predictive_log_rmse_q95": posterior_predictive_log_rmse_q95,
        "validation_observation_indices": np.asarray(validation_observation_indices, dtype=np.int64),
        "validation_mean_l2_delta": validation_mean_l2_delta,
        "validation_cov_fro_delta": validation_cov_fro_delta,
    }
    return observations, metadata, {"theta_samples": theta_samples, **diagnostics}


def sample_glu_reference_posterior(
    *,
    x_full: np.ndarray,
    prior_low: np.ndarray,
    prior_high: np.ndarray,
    simulator_scale: float,
    num_reference_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample p(theta | x) for GLU under a uniform box prior."""
    x = np.asarray(x_full, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError("x_full must have shape (num_observations, theta_dim).")
    prior_low = np.asarray(prior_low, dtype=float)
    prior_high = np.asarray(prior_high, dtype=float)
    if prior_low.shape != (x.shape[1],) or prior_high.shape != (x.shape[1],):
        raise ValueError("prior bounds must match x_full feature dimension.")
    if simulator_scale <= 0:
        raise ValueError("simulator_scale must be positive.")
    if num_reference_samples <= 0:
        raise ValueError("num_reference_samples must be positive.")

    a = (prior_low[None, :] - x) / simulator_scale
    b = (prior_high[None, :] - x) / simulator_scale
    samples = np.empty((x.shape[0], num_reference_samples, x.shape[1]), dtype=float)
    for obs_idx in range(x.shape[0]):
        samples[obs_idx] = truncnorm.rvs(
            a=a[obs_idx],
            b=b[obs_idx],
            loc=x[obs_idx],
            scale=simulator_scale,
            size=(num_reference_samples, x.shape[1]),
            random_state=rng,
        )
    return samples


def sample_glu_mcar_partial_data_reference_posterior(
    *,
    x_obs: np.ndarray,
    mask: np.ndarray,
    prior_low: np.ndarray,
    prior_high: np.ndarray,
    simulator_scale: float,
    num_reference_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample the analytic GLU posterior given observed values and an MCAR mask.

    GLU coordinates are conditionally independent. For an observed coordinate,
    the posterior is its usual prior-truncated Normal centered at ``x_obs``;
    for a missing coordinate, the observation likelihood integrates to one and
    the posterior is the original uniform prior.
    """
    x = np.asarray(x_obs, dtype=float)
    mask_array = np.asarray(mask)
    if x.ndim == 1:
        x = x[None, :]
    if mask_array.ndim == 1:
        mask_array = mask_array[None, :]
    if x.ndim != 2 or mask_array.shape != x.shape:
        raise ValueError("x_obs and mask must have matching shape (num_observations, theta_dim).")
    if not np.all((mask_array == 0) | (mask_array == 1)):
        raise ValueError("mask must be binary with 1=observed and 0=missing.")
    prior_low = np.asarray(prior_low, dtype=float)
    prior_high = np.asarray(prior_high, dtype=float)
    if prior_low.shape != (x.shape[1],) or prior_high.shape != (x.shape[1],):
        raise ValueError("prior bounds must match x_obs feature dimension.")
    if np.any(prior_high <= prior_low):
        raise ValueError("prior_high must be greater than prior_low.")
    if simulator_scale <= 0:
        raise ValueError("simulator_scale must be positive.")
    if num_reference_samples <= 0:
        raise ValueError("num_reference_samples must be positive.")

    # Retain exact existing sampler behavior when nothing is missing.
    if np.all(mask_array == 1):
        return sample_glu_reference_posterior(
            x_full=x,
            prior_low=prior_low,
            prior_high=prior_high,
            simulator_scale=simulator_scale,
            num_reference_samples=num_reference_samples,
            rng=rng,
        )

    samples = rng.uniform(
        low=prior_low,
        high=prior_high,
        size=(x.shape[0], num_reference_samples, x.shape[1]),
    )
    for obs_idx, coordinate in np.argwhere(mask_array == 1):
        a = (prior_low[coordinate] - x[obs_idx, coordinate]) / simulator_scale
        b = (prior_high[coordinate] - x[obs_idx, coordinate]) / simulator_scale
        samples[obs_idx, :, coordinate] = truncnorm.rvs(
            a=a,
            b=b,
            loc=x[obs_idx, coordinate],
            scale=simulator_scale,
            size=num_reference_samples,
            random_state=rng,
        )
    return samples


def glu_mnar_mask_log_probability(
    x_full: np.ndarray, mask: np.ndarray, *, missing_fraction: float, eps: float = 1e-12
) -> float:
    """Exact log p(mask | x_full) for MeanNormalizedSelfCensoringMNARMask.

    This mirrors ``MeanNormalizedSelfCensoringMNARMask`` rather than duplicating
    an approximate version of its score law.  Inputs here are one observation.
    """
    x = np.asarray(x_full, dtype=float)
    m = np.asarray(mask, dtype=np.int8)
    if x.ndim != 1 or m.shape != x.shape or not np.all((m == 0) | (m == 1)):
        raise ValueError("x_full and mask must be matching one-dimensional binary-mask arrays.")
    generator = MeanNormalizedSelfCensoringMNARMask(missing_fraction=missing_fraction, eps=eps)
    score = generator._score(generator._transform_values(x))
    p_missing = np.clip(missing_fraction * score, 0.0, 1.0)
    probabilities = np.where(m == 0, p_missing, 1.0 - p_missing)
    if np.any(probabilities == 0.0):
        return float("-inf")
    return float(np.log(probabilities).sum())


def glu_prior_predictive_logpdf(x: np.ndarray, *, prior_low: np.ndarray, prior_high: np.ndarray, simulator_scale: float) -> np.ndarray:
    """Independent GLU prior-predictive log density for observation coordinates."""
    value = np.asarray(x, dtype=float)
    low, high = np.asarray(prior_low, dtype=float), np.asarray(prior_high, dtype=float)
    log_hi = log_ndtr((high - value) / simulator_scale)
    log_lo = log_ndtr((low - value) / simulator_scale)
    # log(exp(log_hi)-exp(log_lo)), stable in both tails.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_mass = log_hi + np.log1p(-np.exp(np.minimum(log_lo - log_hi, 0.0)))
    return log_mass - np.log(high - low)


def _glu_prior_predictive_draw(rng: np.random.Generator, n: int, low: np.ndarray, high: np.ndarray, scale: float) -> np.ndarray:
    return rng.uniform(low, high, size=(n, low.size)) + rng.normal(0.0, scale, size=(n, low.size))


def sample_glu_mnar_missing_emcee(
    *, x_obs: np.ndarray, mask: np.ndarray, prior_low: np.ndarray, prior_high: np.ndarray,
    simulator_scale: float, missing_fraction: float, num_samples: int, rng: np.random.Generator,
    nwalkers: int | None = None, burnin: int = 500, steps: int = 1_000,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample latent missing GLU observations with an emcee ensemble."""
    import emcee
    x, m = np.asarray(x_obs, float).copy(), np.asarray(mask, np.int8)
    missing = np.flatnonzero(m == 0)
    if not missing.size:
        return np.empty((num_samples, 0)), {"acceptance_fraction": 1.0, "autocorr_time": np.empty(0), "converged": True}
    low, high = np.asarray(prior_low, float), np.asarray(prior_high, float)
    def log_target(z: np.ndarray) -> float:
        full = x.copy(); full[missing] = z
        return float(glu_prior_predictive_logpdf(z, prior_low=low[missing], prior_high=high[missing], simulator_scale=simulator_scale).sum() + glu_mnar_mask_log_probability(full, m, missing_fraction=missing_fraction))
    ndim = missing.size; nwalkers = max(2 * ndim + 4, 16) if nwalkers is None else nwalkers
    initial = []
    while len(initial) < nwalkers:
        candidate = _glu_prior_predictive_draw(rng, 1, low[missing], high[missing], simulator_scale)[0]
        if np.isfinite(log_target(candidate)): initial.append(candidate)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_target)
    state = sampler.run_mcmc(np.asarray(initial), burnin, progress=False)
    sampler.reset(); sampler.run_mcmc(state, steps, progress=False)
    chain = sampler.get_chain(flat=True)
    draws = chain[rng.choice(len(chain), size=num_samples, replace=len(chain) < num_samples)]
    try:
        tau = sampler.get_autocorr_time(tol=0)
        converged = bool(steps >= 50 * np.max(tau))
    except Exception:
        tau = np.full(ndim, np.nan); converged = False
    return draws, {"acceptance_fraction": float(np.mean(sampler.acceptance_fraction)), "autocorr_time": tau,
                   "emcee_ess": np.where(np.isfinite(tau), len(chain) / tau, np.nan), "converged": converged,
                   "nwalkers": nwalkers, "burnin": burnin, "steps": steps,
                   # Walkers interact and are not chains; this mean trace is only
                   # for comparing separately seeded ensembles.
                   "ensemble_mean_trace": sampler.get_chain().mean(axis=1)}


def sample_glu_mnar_missing_independence_mh(
    *, x_obs: np.ndarray, mask: np.ndarray, prior_low: np.ndarray, prior_high: np.ndarray,
    simulator_scale: float, missing_fraction: float, num_samples: int, rng: np.random.Generator,
    burnin: int = 2_000, thin: int = 5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Independent-coordinate MH; proposals are the exact GLU prior predictive."""
    x, m = np.asarray(x_obs, float).copy(), np.asarray(mask, np.int8); missing = np.flatnonzero(m == 0)
    if not missing.size: return np.empty((num_samples, 0)), {"acceptance_fraction": 1.0}
    low, high = np.asarray(prior_low, float), np.asarray(prior_high, float)
    def lp(z):
        full=x.copy(); full[missing]=z; return glu_mnar_mask_log_probability(full,m,missing_fraction=missing_fraction)
    current = _glu_prior_predictive_draw(rng, 1, low[missing], high[missing], simulator_scale)[0]
    while not np.isfinite(lp(current)): current = _glu_prior_predictive_draw(rng, 1, low[missing], high[missing], simulator_scale)[0]
    accepted=0; out=[]
    for t in range(burnin + num_samples * thin):
        proposal=current.copy(); j=rng.integers(missing.size)
        proposal[j]=_glu_prior_predictive_draw(rng, 1, low[missing[j:j+1]], high[missing[j:j+1]], simulator_scale)[0,0]
        if np.log(rng.uniform()) < lp(proposal)-lp(current): current=proposal; accepted+=1
        if t >= burnin and (t-burnin) % thin == 0: out.append(current.copy())
    return np.asarray(out), {"acceptance_fraction": accepted / (burnin + num_samples * thin), "burnin": burnin, "thin": thin}


def sample_glu_mnar_missing_importance(
    *, x_obs: np.ndarray, mask: np.ndarray, prior_low: np.ndarray, prior_high: np.ndarray,
    simulator_scale: float, missing_fraction: float, num_proposals: int, num_samples: int, rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Prior-predictive importance sampler, intended as a validation baseline."""
    x, m=np.asarray(x_obs,float).copy(),np.asarray(mask,np.int8); missing=np.flatnonzero(m==0)
    proposals=_glu_prior_predictive_draw(rng,num_proposals,np.asarray(prior_low)[missing],np.asarray(prior_high)[missing],simulator_scale)
    logw=np.empty(num_proposals)
    for i,z in enumerate(proposals): full=x.copy(); full[missing]=z; logw[i]=glu_mnar_mask_log_probability(full,m,missing_fraction=missing_fraction)
    logw-=logsumexp(logw); w=np.exp(logw); ess=1/np.sum(w*w)
    return proposals[rng.choice(num_proposals,size=num_samples,p=w)], {"ess": float(ess), "ess_fraction": float(ess/num_proposals), "max_weight":float(w.max())}


def sample_glu_mnar_partial_data_reference_posterior(**kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Draw theta samples by emcee latent-x sampling then exact truncated-Normal theta draws."""
    x_obs=np.asarray(kwargs["x_obs"],float); mask=np.asarray(kwargs["mask"],np.int8)
    if x_obs.ndim != 1: raise ValueError("This pilot sampler accepts one observation at a time.")
    rng=kwargs["rng"]; n=int(kwargs["num_reference_samples"])
    latent, diag=sample_glu_mnar_missing_emcee(x_obs=x_obs,mask=mask,prior_low=kwargs["prior_low"],prior_high=kwargs["prior_high"],simulator_scale=kwargs["simulator_scale"],missing_fraction=kwargs["missing_fraction"],num_samples=n,rng=rng,burnin=kwargs.get("burnin",500),steps=kwargs.get("steps",1000))
    full=np.repeat(x_obs[None,:],n,axis=0); full[:,mask==0]=latent
    theta=sample_glu_reference_posterior(x_full=full,prior_low=kwargs["prior_low"],prior_high=kwargs["prior_high"],simulator_scale=kwargs["simulator_scale"],num_reference_samples=1,rng=rng)[:,0,:]
    return theta,diag


def glm_log_likelihood(
    theta: np.ndarray,
    x_full: np.ndarray,
    simulator: GLMSimulator | None = None,
) -> np.ndarray:
    """Evaluate the raw Bernoulli GLM log likelihood."""
    simulator = (
        GLMSimulator(summary="raw") if simulator is None else simulator
    )
    if simulator.summary != "raw":
        raise ValueError("GLM reference likelihood expects a raw GLMSimulator.")
    theta_array = np.asarray(theta, dtype=float)
    single = theta_array.ndim == 1
    if single:
        theta_array = theta_array[None, :]
    if theta_array.ndim != 2 or theta_array.shape[1] != simulator.theta_dim:
        raise ValueError(
            f"theta must have shape ({simulator.theta_dim},) or (batch, {simulator.theta_dim})."
        )
    x = np.asarray(x_full, dtype=float)
    if x.shape != simulator.x_shape:
        raise ValueError(f"x_full must have shape {simulator.x_shape}.")
    if not np.all((x == 0) | (x == 1)):
        raise ValueError("raw GLM x_full must be binary.")

    psi = theta_array @ simulator.design_matrix.T
    log_like = np.sum(x[None, :] * psi - np.logaddexp(0.0, psi), axis=1)
    return log_like[0] if single else log_like


def glm_mnar_mask_log_probability(
    x_full: np.ndarray,
    mask: np.ndarray,
    *,
    missing_fraction: float,
    eps: float = 1e-12,
) -> float:
    """Exact log ``p(mask | x_full)`` for binary mean-normalized self-censoring."""
    x = np.asarray(x_full, dtype=float)
    m = np.asarray(mask, dtype=np.int8)
    if x.ndim != 1 or m.shape != x.shape or not np.all((x == 0) | (x == 1)):
        raise ValueError("x_full must be binary and mask must have matching shape.")
    if not np.all((m == 0) | (m == 1)) or not 0.0 <= missing_fraction <= 1.0:
        raise ValueError("mask must be binary and missing_fraction must be in [0, 1].")
    if np.max(x) - np.min(x) < eps:
        score = np.full_like(x, 0.5) / (0.5 + eps)
    else:
        score = (x - np.min(x)) / (np.max(x) - np.min(x) + eps)
        score = score / (np.mean(score) + eps)
    p = np.clip(missing_fraction * score, 0.0, 1.0)
    missing = m == 0
    if np.any(missing & (p == 0.0)) or np.any(~missing & (p == 1.0)):
        return float(-np.inf)
    return float(np.log(p[missing]).sum() + np.log1p(-p[~missing]).sum())


def glm_mnar_compatible_completions(
    x_obs: np.ndarray,
    mask: np.ndarray,
    *,
    missing_fraction: float,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate all binary completions with nonzero MNAR mask probability.

    For this binary mechanism there are at most two: a nonconstant completion
    with every missing value equal to one, and (when compatible) a constant
    all-zero or all-one completion.
    """
    x = np.asarray(x_obs, dtype=float)
    m = np.asarray(mask, dtype=np.int8)
    if x.ndim != 1 or x.shape != m.shape or not np.all((m == 0) | (m == 1)):
        raise ValueError("x_obs and binary mask must be matching one-dimensional arrays.")
    observed = m.astype(bool)
    if not np.all((x[observed] == 0) | (x[observed] == 1)):
        raise ValueError("observed values must be binary.")
    candidates: list[np.ndarray] = []
    if np.all(observed):
        candidates.append(x.copy())
    else:
        if not np.any(observed & (x == 1)):
            candidates.append(np.zeros_like(x))
        if not np.any(observed & (x == 0)):
            candidates.append(np.ones_like(x))
        if np.any(observed & (x == 0)):
            mixed = x.copy()
            mixed[~observed] = 1.0
            candidates.append(mixed)
    kept = [(c, glm_mnar_mask_log_probability(c, m, missing_fraction=missing_fraction, eps=eps)) for c in candidates]
    kept = [(c, lp) for c, lp in kept if np.isfinite(lp)]
    return np.asarray([c for c, _ in kept]), np.asarray([lp for _, lp in kept])


def glm_mnar_log_marginal_likelihood(
    theta: np.ndarray,
    x_obs: np.ndarray,
    mask: np.ndarray,
    *,
    missing_fraction: float,
    simulator: GLMSimulator | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Exact finite-sum GLM/MNAR likelihood after marginalizing binary outcomes."""
    simulator = GLMSimulator(summary="raw") if simulator is None else simulator
    completions, log_mask = glm_mnar_compatible_completions(
        x_obs, mask, missing_fraction=missing_fraction, eps=eps
    )
    theta_array = np.asarray(theta, dtype=float)
    single = theta_array.ndim == 1
    if single:
        theta_array = theta_array[None, :]
    terms = np.stack([glm_log_likelihood(theta_array, c, simulator) + lp for c, lp in zip(completions, log_mask)], axis=0)
    result = logsumexp(terms, axis=0)
    return result[0] if single else result


def glm_observed_log_likelihood(
    theta: np.ndarray,
    x_obs: np.ndarray,
    mask: np.ndarray,
    simulator: GLMSimulator | None = None,
) -> np.ndarray:
    """Raw GLM log likelihood restricted to coordinates marked observed.

    ``mask == 1`` denotes an observed Bernoulli measurement.  Values in
    missing coordinates are intentionally neither inspected nor used, so a
    zero-imputed ``x_obs`` cannot be mistaken for observed data.
    """
    simulator = GLMSimulator(summary="raw") if simulator is None else simulator
    if simulator.summary != "raw":
        raise ValueError("GLM reference likelihood expects a raw GLMSimulator.")
    theta_array = np.asarray(theta, dtype=float)
    single = theta_array.ndim == 1
    if single:
        theta_array = theta_array[None, :]
    if theta_array.ndim != 2 or theta_array.shape[1] != simulator.theta_dim:
        raise ValueError(
            f"theta must have shape ({simulator.theta_dim},) or (batch, {simulator.theta_dim})."
        )
    x = np.asarray(x_obs)
    observed = np.asarray(mask)
    if x.shape != simulator.x_shape or observed.shape != simulator.x_shape:
        raise ValueError(f"x_obs and mask must have shape {simulator.x_shape}.")
    if not np.all((observed == 0) | (observed == 1)):
        raise ValueError("mask must be binary with 1=observed and 0=missing.")
    observed = observed.astype(bool, copy=False)
    if not np.all((x[observed] == 0) | (x[observed] == 1)):
        raise ValueError("observed raw GLM measurements must be binary.")
    if not np.any(observed):
        result = np.zeros(theta_array.shape[0], dtype=float)
        return result[0] if single else result
    design = simulator.design_matrix[observed]
    x_observed = np.asarray(x[observed], dtype=float)
    psi = theta_array @ design.T
    log_like = np.sum(x_observed[None, :] * psi - np.logaddexp(0.0, psi), axis=1)
    return log_like[0] if single else log_like


def glm_observed_log_prob(
    theta: np.ndarray,
    x_obs: np.ndarray,
    mask: np.ndarray,
    simulator: GLMSimulator,
) -> np.ndarray:
    """Log ``p(theta | x_obs, mask)`` for GLM under ignorable MCAR masking."""
    theta_array = np.asarray(theta, dtype=float)
    single = theta_array.ndim == 1
    if single:
        theta_array = theta_array[None, :]
    log_prob = simulator.log_prior(theta_array)
    finite = np.isfinite(log_prob)
    if np.any(finite):
        log_prob[finite] += glm_observed_log_likelihood(
            theta_array[finite], x_obs, mask, simulator
        )
    return log_prob[0] if single else log_prob


def glm_log_prob(
    theta: np.ndarray,
    x_full: np.ndarray,
    simulator: GLMSimulator,
) -> np.ndarray:
    """Evaluate the raw GLM log posterior up to an additive constant."""
    theta_array = np.asarray(theta, dtype=float)
    single = theta_array.ndim == 1
    if single:
        theta_array = theta_array[None, :]
    log_prob = simulator.log_prior(theta_array)
    finite = np.isfinite(log_prob)
    if np.any(finite):
        log_prob[finite] += glm_log_likelihood(
            theta_array[finite],
            x_full,
            simulator,
        )
    return log_prob[0] if single else log_prob


def run_glm_emcee_reference(
    x_full: np.ndarray,
    *,
    simulator: GLMSimulator,
    num_walkers: int,
    burn_in_steps: int,
    production_steps: int,
    initial_scale: float,
    rng: np.random.Generator,
    emcee_seed: int | None = None,
) -> dict[str, np.ndarray | float]:
    """Run one vectorized emcee ensemble for a raw GLM observation."""
    import emcee

    map_estimate, map_log_prob = find_glm_map(x_full, simulator)
    initial_state = initialize_glm_walkers(
        center=map_estimate,
        simulator=simulator,
        num_walkers=num_walkers,
        scale=initial_scale,
        rng=rng,
    )

    def log_prob_fn(theta_batch: np.ndarray) -> np.ndarray:
        return glm_log_prob(theta_batch, x_full, simulator)

    sampler = emcee.EnsembleSampler(
        num_walkers,
        simulator.theta_dim,
        log_prob_fn,
        vectorize=True,
    )
    # emcee otherwise initializes its private RandomState from NumPy's global
    # legacy RNG.  Supplying this optional seed makes a reference run fully
    # reproducible without mutating global process state.
    if emcee_seed is not None:
        sampler.random_state = np.random.RandomState(int(emcee_seed)).get_state()
    sampler.run_mcmc(initial_state, burn_in_steps, progress=False)
    sampler.reset()
    sampler.run_mcmc(None, production_steps, progress=False)
    chain = sampler.get_chain()
    acceptance_fraction_by_walker = np.asarray(sampler.acceptance_fraction, dtype=float)
    acceptance_fraction = float(np.mean(acceptance_fraction_by_walker))
    try:
        autocorr_time = sampler.get_autocorr_time(tol=0)
        emcee_ess = num_walkers * production_steps / autocorr_time
    except Exception:
        autocorr_time = np.full(simulator.theta_dim, np.nan, dtype=float)
        emcee_ess = np.full(simulator.theta_dim, np.nan, dtype=float)
    split_rhat = compute_split_rhat(chain)
    (
        walker_group_mean_l2_max,
        walker_group_std_l2_max,
    ) = compute_walker_group_agreement(chain)
    return {
        "chain": chain,
        "acceptance_fraction": acceptance_fraction,
        "acceptance_fraction_by_walker": acceptance_fraction_by_walker,
        "autocorr_time": autocorr_time,
        "emcee_ess": emcee_ess,
        "split_rhat": split_rhat,
        "walker_group_mean_l2_max": float(walker_group_mean_l2_max),
        "walker_group_std_l2_max": float(walker_group_std_l2_max),
        "map_estimate": map_estimate,
        "map_log_prob": float(map_log_prob),
    }


def find_glm_map(
    x_full: np.ndarray,
    simulator: GLMSimulator,
) -> tuple[np.ndarray, float]:
    """Find a GLM posterior mode for walker initialization."""
    starts = [
        simulator.prior_mean,
        simulator.prior_mean + 0.5 * np.sqrt(np.diag(simulator.prior_covariance)),
        simulator.prior_mean - 0.5 * np.sqrt(np.diag(simulator.prior_covariance)),
    ]

    def objective(theta: np.ndarray) -> float:
        return -float(glm_log_prob(theta, x_full, simulator))

    best_x = starts[0]
    best_fun = objective(best_x)
    for start in starts:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
        )
        if result.success and float(result.fun) < best_fun:
            best_x = np.asarray(result.x, dtype=float)
            best_fun = float(result.fun)
    return best_x, -best_fun


def initialize_glm_walkers(
    *,
    center: np.ndarray,
    simulator: GLMSimulator,
    num_walkers: int,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Initialize GLM walkers near a mode with finite Gaussian prior density."""
    walkers = rng.normal(
        loc=center,
        scale=scale,
        size=(num_walkers, simulator.theta_dim),
    )
    invalid = ~np.isfinite(simulator.log_prior(walkers))
    while np.any(invalid):
        walkers[invalid] = rng.normal(
            loc=center,
            scale=scale,
            size=(int(np.sum(invalid)), simulator.theta_dim),
        )
        invalid = ~np.isfinite(simulator.log_prior(walkers))
    return walkers


def find_glm_observed_map(
    x_obs: np.ndarray,
    mask: np.ndarray,
    simulator: GLMSimulator,
) -> tuple[np.ndarray, float]:
    """Find the unique GLM observed-data posterior mode and its log density."""
    observed = np.asarray(mask, dtype=bool)
    x = np.asarray(x_obs, dtype=float)
    if x.shape != simulator.x_shape or observed.shape != simulator.x_shape:
        raise ValueError(f"x_obs and mask must have shape {simulator.x_shape}.")
    if not np.all((x[observed] == 0) | (x[observed] == 1)):
        raise ValueError("observed raw GLM measurements must be binary.")
    design = simulator.design_matrix[observed]
    y = x[observed]

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        centered = theta - simulator.prior_mean
        if design.shape[0]:
            psi = design @ theta
            p = expit(psi)
            log_like = float(np.sum(y * psi - np.logaddexp(0.0, psi)))
            gradient_like = design.T @ (y - p)
        else:
            log_like = 0.0
            gradient_like = np.zeros(simulator.theta_dim, dtype=float)
        log_prior = float(simulator.log_prior(theta))
        log_prob = log_prior + log_like
        gradient = -simulator.prior_precision @ centered + gradient_like
        return -log_prob, -gradient

    starts = (
        simulator.prior_mean,
        simulator.prior_mean + 0.5 * np.sqrt(np.diag(simulator.prior_covariance)),
        simulator.prior_mean - 0.5 * np.sqrt(np.diag(simulator.prior_covariance)),
    )
    results = [
        minimize(lambda z: objective(z)[0], start, jac=lambda z: objective(z)[1], method="L-BFGS-B")
        for start in starts
    ]
    successful = [r for r in results if r.success and np.isfinite(r.fun)]
    result = min(successful or results, key=lambda r: float(r.fun))
    if not np.isfinite(result.fun):
        raise RuntimeError("Unable to find a finite GLM observed-data posterior mode.")
    mode = np.asarray(result.x, dtype=float)
    return mode, float(glm_observed_log_prob(mode, x_obs, mask, simulator))


def glm_observed_laplace_covariance(
    mode: np.ndarray,
    x_obs: np.ndarray,
    mask: np.ndarray,
    simulator: GLMSimulator,
) -> np.ndarray:
    """Inverse negative Hessian at the observed-data GLM posterior mode."""
    observed = np.asarray(mask, dtype=bool)
    design = simulator.design_matrix[observed]
    if design.shape[0] == 0:
        return simulator.prior_covariance.copy()
    p = expit(design @ np.asarray(mode, dtype=float))
    negative_hessian = simulator.prior_precision + design.T @ (
        (p * (1.0 - p))[:, None] * design
    )
    try:
        return np.linalg.inv(negative_hessian)
    except np.linalg.LinAlgError as error:
        raise RuntimeError("Observed-data GLM Laplace precision is not invertible.") from error


def run_glm_observed_emcee_reference(
    x_obs: np.ndarray,
    mask: np.ndarray,
    *,
    simulator: GLMSimulator,
    num_walkers: int,
    burn_in_steps: int,
    production_steps: int,
    initial_scale: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray | float]:
    """Run an emcee ensemble for an ignorable-MCAR, partially observed GLM."""
    import emcee

    if num_walkers < 2 * simulator.theta_dim:
        raise ValueError("num_walkers must be at least twice the GLM dimension.")
    mode, mode_log_prob = find_glm_observed_map(x_obs, mask, simulator)
    initial_state = initialize_glm_walkers(
        center=mode, simulator=simulator, num_walkers=num_walkers,
        scale=initial_scale, rng=rng,
    )

    def log_prob_fn(theta_batch: np.ndarray) -> np.ndarray:
        return glm_observed_log_prob(theta_batch, x_obs, mask, simulator)

    sampler = emcee.EnsembleSampler(num_walkers, simulator.theta_dim, log_prob_fn, vectorize=True)
    sampler.run_mcmc(initial_state, burn_in_steps, progress=False)
    sampler.reset()
    sampler.run_mcmc(None, production_steps, progress=False)
    chain = sampler.get_chain()
    try:
        autocorr_time = sampler.get_autocorr_time(tol=0)
        emcee_ess = num_walkers * production_steps / autocorr_time
    except Exception:
        autocorr_time = np.full(simulator.theta_dim, np.nan)
        emcee_ess = np.full(simulator.theta_dim, np.nan)
    return {
        "chain": chain,
        "acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "autocorr_time": autocorr_time,
        "emcee_ess": emcee_ess,
        "split_rhat": compute_split_rhat(chain),
        "map_estimate": mode,
        "map_log_prob": mode_log_prob,
        "laplace_covariance": glm_observed_laplace_covariance(mode, x_obs, mask, simulator),
    }


def run_glm_observed_independence_mh(
    x_obs: np.ndarray,
    mask: np.ndarray,
    *,
    simulator: GLMSimulator,
    num_steps: int,
    burn_in_steps: int,
    thin: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray | float]:
    """Exact independence-MH using the observed-data Laplace proposal.

    This deliberately independent sampler is used for validation.  The proposal
    is Gaussian at the posterior mode; its covariance is the inverse exact
    negative Hessian at that mode.
    """
    if num_steps <= burn_in_steps or thin <= 0:
        raise ValueError("num_steps must exceed burn_in_steps and thin must be positive.")
    mode, mode_log_prob = find_glm_observed_map(x_obs, mask, simulator)
    covariance = glm_observed_laplace_covariance(mode, x_obs, mask, simulator)
    chol = np.linalg.cholesky(covariance)
    log_det_covariance = 2.0 * np.log(np.diag(chol)).sum()
    dimension = simulator.theta_dim

    def log_proposal(theta: np.ndarray) -> float:
        delta = theta - mode
        whitened = np.linalg.solve(chol, delta)
        return float(-0.5 * (dimension * np.log(2.0 * np.pi) + log_det_covariance + whitened @ whitened))

    current = mode.copy()
    current_log_target = mode_log_prob
    current_log_proposal = log_proposal(current)
    chain = np.empty((num_steps, dimension), dtype=float)
    accepted = 0
    for step in range(num_steps):
        proposal = mode + chol @ rng.normal(size=dimension)
        proposal_log_target = float(glm_observed_log_prob(proposal, x_obs, mask, simulator))
        proposal_log_proposal = log_proposal(proposal)
        log_acceptance = ((proposal_log_target - proposal_log_proposal)
                          - (current_log_target - current_log_proposal))
        if np.log(rng.uniform()) < min(0.0, log_acceptance):
            current, current_log_target, current_log_proposal = proposal, proposal_log_target, proposal_log_proposal
            accepted += 1
        chain[step] = current
    return {
        "chain": chain[burn_in_steps::thin],
        "acceptance_fraction": float(accepted / num_steps),
        "map_estimate": mode,
        "map_log_prob": mode_log_prob,
        "laplace_covariance": covariance,
    }


def generate_glm_mcar_partial_data_reference_posteriors(
    *,
    full_reference_path: str | os.PathLike[str] = (
        "references/reference_posteriors_v1/glm_references.h5"
    ),
    missing_fraction: float,
    reference_mask_seed: int = 987_654,
    posterior_seed: int = 23_456,
    num_reference_samples: int = 10_000,
    num_mh_chains: int = 4,
    mh_steps: int = 50_000,
    mh_burn_in_steps: int = 2_000,
    mh_thin: int = 2,
    max_rank_rhat: float = 1.01,
    min_bulk_ess: float = 1_000.0,
    max_attempts: int = 3,
    retry_step_multiplier: float = 2.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, np.ndarray]]:
    """Generate GLM/MCAR samples from ``p(theta | x_obs, mask)``.

    The original full-reference observations are loaded rather than regenerated,
    and the one batched Point-MCAR draw matches the evaluation convention.
    This is a production interface only; callers choose when to write artifacts.
    """
    if num_reference_samples <= 0 or num_mh_chains <= 0 or max_attempts <= 0:
        raise ValueError("num_reference_samples, num_mh_chains, and max_attempts must be positive.")
    if retry_step_multiplier <= 1.0:
        raise ValueError("retry_step_multiplier must exceed one.")
    full_obs, _, full_meta, _ = load_reference_posteriors_hdf5(full_reference_path)
    if full_meta.get("problem") != "glm":
        raise ValueError("full_reference_path must be a GLM reference artifact.")
    sim_meta = full_meta["simulator"]
    simulator = GLMSimulator(
        dim=int(sim_meta["dim"]), duration=int(sim_meta["duration"]),
        stimulus_seed=int(sim_meta["stimulus_seed"]), summary="raw",
    )
    x_full = np.asarray(full_obs["x_full"], dtype=np.int8)
    theta_true = np.asarray(full_obs["theta_true"], dtype=float)
    mask_generator = PointMCARMask(missing_fraction=missing_fraction)
    mask = mask_generator.generate(x_full, theta_true, np.random.default_rng(reference_mask_seed))
    x_obs = x_full * mask
    posterior_rng = np.random.default_rng(posterior_seed)
    theta_samples = np.empty((x_full.shape[0], num_reference_samples, simulator.theta_dim), dtype=float)
    acceptance = np.empty((x_full.shape[0], num_mh_chains), dtype=float)
    map_estimates = np.empty((x_full.shape[0], simulator.theta_dim), dtype=float)
    map_log_prob = np.empty(x_full.shape[0], dtype=float)
    laplace_covariance = np.empty((x_full.shape[0], simulator.theta_dim, simulator.theta_dim), dtype=float)
    retained_counts = np.empty((x_full.shape[0], num_mh_chains), dtype=np.int64)
    rank_rhat = np.empty((x_full.shape[0], simulator.theta_dim), dtype=float)
    bulk_ess = np.empty_like(rank_rhat)
    tail_ess = np.empty_like(rank_rhat)
    runtime_seconds = np.empty(x_full.shape[0], dtype=float)
    mh_steps_used = np.empty(x_full.shape[0], dtype=np.int64)
    mh_attempts = np.empty(x_full.shape[0], dtype=np.int64)
    import arviz as az

    for index in range(x_full.shape[0]):
        started = time.perf_counter()
        steps_this_attempt, burn_this_attempt = mh_steps, mh_burn_in_steps
        for attempt in range(1, max_attempts + 1):
            chains, details = [], []
            for _ in range(num_mh_chains):
                result = run_glm_observed_independence_mh(
                    x_obs[index], mask[index], simulator=simulator, num_steps=steps_this_attempt,
                    burn_in_steps=burn_this_attempt, thin=mh_thin,
                    rng=np.random.default_rng(posterior_rng.integers(0, np.iinfo(np.uint32).max)),
                )
                chains.append(np.asarray(result["chain"]))
                details.append(result)
            idata = az.from_dict(posterior={"theta": np.stack(chains)})
            rhat_this = az.rhat(idata, method="rank").theta.values
            bulk_this = az.ess(idata, method="bulk").theta.values
            tail_this = az.ess(idata, method="tail").theta.values
            if np.nanmax(rhat_this) <= max_rank_rhat and np.nanmin(bulk_this) >= min_bulk_ess:
                break
            if attempt == max_attempts:
                raise RuntimeError(
                    f"GLM MCAR observation {index} failed MH convergence gates after {attempt} attempts: "
                    f"max rank-Rhat={np.nanmax(rhat_this):.4f}, min bulk ESS={np.nanmin(bulk_this):.1f}."
                )
            steps_this_attempt = int(np.ceil(steps_this_attempt * retry_step_multiplier))
            burn_this_attempt = int(np.ceil(burn_this_attempt * retry_step_multiplier))
        acceptance[index] = [float(result["acceptance_fraction"]) for result in details]
        retained_counts[index] = [chain.shape[0] for chain in chains]
        map_estimates[index] = np.asarray(details[0]["map_estimate"])
        map_log_prob[index] = float(details[0]["map_log_prob"])
        laplace_covariance[index] = np.asarray(details[0]["laplace_covariance"])
        rank_rhat[index], bulk_ess[index], tail_ess[index] = rhat_this, bulk_this, tail_this
        mh_steps_used[index], mh_attempts[index] = steps_this_attempt, attempt
        pooled = np.concatenate(chains, axis=0)
        draw_indices = posterior_rng.choice(
            pooled.shape[0], size=num_reference_samples, replace=pooled.shape[0] < num_reference_samples
        )
        theta_samples[index] = pooled[draw_indices]
        runtime_seconds[index] = time.perf_counter() - started
    observations = {"theta_true": theta_true, "x_full": x_full, "x_obs": x_obs, "mask": mask}
    metadata = {
        "problem": "glm", "missingness_mechanism": "mcar",
        "posterior_method": "laplace_independence_metropolis_hastings",
        "posterior_conditioning": "theta_given_x_obs_mask",
        "mask": mask_generator.metadata(), "reference_mask_seed": int(reference_mask_seed),
        "posterior_seed": int(posterior_seed), "full_data_reference_path": str(full_reference_path),
        "full_data_reference_version": full_meta.get("reference_version"),
        "simulator": simulator.metadata(), "theta_scaled": False, "x_scaled": False,
        "num_mh_chains": int(num_mh_chains), "mh_steps": int(mh_steps),
        "mh_burn_in_steps": int(mh_burn_in_steps), "mh_thin": int(mh_thin),
        "max_rank_rhat": float(max_rank_rhat), "min_bulk_ess": float(min_bulk_ess),
        "max_attempts": int(max_attempts), "retry_step_multiplier": float(retry_step_multiplier),
    }
    diagnostics = {
        "posterior_sample_mean": theta_samples.mean(axis=1),
        "posterior_sample_std": theta_samples.std(axis=1),
        "acceptance_fraction": acceptance, "retained_draws_per_chain": retained_counts,
        "rank_rhat": rank_rhat, "bulk_ess": bulk_ess, "tail_ess": tail_ess,
        "runtime_seconds": runtime_seconds, "mh_steps_used": mh_steps_used, "mh_attempts": mh_attempts,
        "map_estimate": map_estimates, "map_log_prob": map_log_prob,
        "laplace_covariance": laplace_covariance,
        "num_nonfinite_posterior_samples": np.array(np.count_nonzero(~np.isfinite(theta_samples)), dtype=np.int64),
    }
    return observations, metadata, {"theta_samples": theta_samples, **diagnostics}


def lotka_volterra_log_likelihood(
    log_theta: np.ndarray,
    x_full: np.ndarray,
    simulator: LotkaVolterraSimulator | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the LV lognormal likelihood, optionally restricted to ``mask``.

    ``mask`` uses GapSBI's convention (one is observed).  This is the exact
    MCAR observed-data likelihood: the deterministic ODE is still solved over
    the complete interval, while conditionally independent lognormal factors
    for missing coordinates are marginalized by omission.
    """
    simulator = LotkaVolterraSimulator() if simulator is None else simulator
    log_theta_array = np.asarray(log_theta, dtype=float)
    single = log_theta_array.ndim == 1
    if single:
        log_theta_array = log_theta_array[None, :]
    if log_theta_array.ndim != 2 or log_theta_array.shape[1] != simulator.theta_dim:
        raise ValueError(
            f"log_theta must have shape ({simulator.theta_dim},) or "
            f"(batch, {simulator.theta_dim})."
        )
    x = np.asarray(x_full, dtype=float)
    if x.shape != simulator.x_shape:
        raise ValueError(f"x_full must have shape {simulator.x_shape}.")
    if mask is None:
        observed = np.ones(simulator.x_shape, dtype=bool)
    else:
        mask_array = np.asarray(mask)
        if mask_array.shape != simulator.x_shape:
            raise ValueError(f"mask must have shape {simulator.x_shape}.")
        if not np.all((mask_array == 0) | (mask_array == 1)):
            raise ValueError("Lotka-Volterra mask entries must be zero or one.")
        observed = mask_array.astype(bool)
    # Zero-imputed missing entries are not measurements and need not be
    # positive.  Observed lognormal measurements must remain strictly so.
    if np.any(x[observed] <= 0):
        raise ValueError("Observed Lotka-Volterra values must be positive.")
    if simulator.observation_noise_scale <= 0:
        raise ValueError("LV likelihood requires positive observation_noise_scale.")

    log_like = np.full(log_theta_array.shape[0], -np.inf, dtype=float)
    # With no observations the likelihood is exactly one, and no ODE solve is
    # needed.  This ensures the masked posterior is the original log prior.
    if not np.any(observed):
        log_like[np.all(np.isfinite(log_theta_array), axis=1)] = 0.0
        return log_like[0] if single else log_like
    for idx, log_theta_i in enumerate(log_theta_array):
        if not np.all(np.isfinite(log_theta_i)):
            continue
        theta_i = np.exp(log_theta_i)
        try:
            states = simulator._solve_states(theta_i)
        except Exception:
            continue
        mean_log = np.log(np.clip(states.T.reshape(-1), 1e-10, simulator.max_state))
        log_like[idx] = np.sum(
            norm.logpdf(
                np.log(x[observed]),
                loc=mean_log[observed],
                scale=simulator.observation_noise_scale,
            )
            - np.log(x[observed])
        )
    return log_like[0] if single else log_like


def lotka_volterra_log_prob(
    log_theta: np.ndarray,
    x_full: np.ndarray,
    simulator: LotkaVolterraSimulator,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the LV log posterior over z=log(theta), up to a constant."""
    log_theta_array = np.asarray(log_theta, dtype=float)
    single = log_theta_array.ndim == 1
    if single:
        log_theta_array = log_theta_array[None, :]
    log_prob = np.full(log_theta_array.shape[0], -np.inf, dtype=float)
    finite = np.all(np.isfinite(log_theta_array), axis=1)
    if np.any(finite):
        prior = np.sum(
            norm.logpdf(
                log_theta_array[finite],
                loc=simulator.prior_log_mean,
                scale=simulator.prior_log_std,
            ),
            axis=1,
        )
        log_prob[finite] = prior + lotka_volterra_log_likelihood(
            log_theta_array[finite],
            x_full,
            simulator,
            mask=mask,
        )
    return log_prob[0] if single else log_prob


def lotka_volterra_log_total_mnar_log_probability(
    x_complete: np.ndarray, mask: np.ndarray, *, missing_fraction: float, eps: float = 1e-12,
) -> float:
    """Exact log p(mask | x) for LV's joint-time log-total MNAR generator."""
    x = np.asarray(x_complete, dtype=float).reshape(-1, 2)
    time_mask = np.asarray(mask).reshape(-1, 2)
    if x.shape != time_mask.shape or not np.all((time_mask == 0) | (time_mask == 1)):
        raise ValueError("LV complete observation and binary mask must have matching interleaved shape.")
    if not np.all(time_mask[:, 0] == time_mask[:, 1]):
        return -np.inf
    if not np.all(np.isfinite(x)) or np.any(x <= 0):
        return -np.inf
    generator = LotkaVolterraLogTotalMNARMask(missing_fraction, eps=eps)
    totals = np.log(x.sum(axis=1) + eps)
    p = np.clip(missing_fraction * generator._score_one(totals), 0.0, 1.0)
    observed = time_mask[:, 0].astype(bool)
    if np.any((~observed) & (p == 0.0)) or np.any(observed & (p == 1.0)):
        return -np.inf
    # Indexing rather than np.where avoids evaluating log(0) on impossible
    # branches that have already been ruled out above.
    return float(np.log1p(-p[observed]).sum() + np.log(p[~observed]).sum())


def lotka_volterra_log_total_mnar_soft_log_probability(
    x_complete: np.ndarray,
    mask: np.ndarray,
    *,
    missing_fraction: float,
    delta: float,
    eps: float = 1e-12,
) -> float:
    """Log softened LV/MNAR mask likelihood used only for an SMC bridge.

    ``delta=0`` is exactly the generator likelihood, including its hard
    support.  For ``0 < delta < .5``, each Bernoulli probability is mapped to
    delta + (1 - 2 delta) p, so every binary time mask has positive mass.
    """
    if delta == 0.0:
        return lotka_volterra_log_total_mnar_log_probability(
            x_complete, mask, missing_fraction=missing_fraction, eps=eps
        )
    if not 0.0 < delta < 0.5:
        raise ValueError("delta must be zero or lie strictly between zero and .5.")
    x = np.asarray(x_complete, dtype=float).reshape(-1, 2)
    time_mask = np.asarray(mask).reshape(-1, 2)
    if x.shape != time_mask.shape or not np.all((time_mask == 0) | (time_mask == 1)):
        raise ValueError("LV complete observation and binary mask must have matching interleaved shape.")
    if not np.all(time_mask[:, 0] == time_mask[:, 1]) or not np.all(np.isfinite(x)) or np.any(x <= 0):
        return -np.inf
    generator = LotkaVolterraLogTotalMNARMask(missing_fraction, eps=eps)
    totals = np.log(x.sum(axis=1) + eps)
    probability = np.clip(missing_fraction * generator._score_one(totals), 0.0, 1.0)
    softened = delta + (1.0 - 2.0 * delta) * probability
    observed = time_mask[:, 0].astype(bool)
    return float(np.log1p(-softened[observed]).sum() + np.log(softened[~observed]).sum())


def sample_lotka_volterra_mnar_missing_observations(
    log_theta: np.ndarray, x_obs: np.ndarray, mask: np.ndarray, *, simulator: LotkaVolterraSimulator,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw latent LV noisy measurements exactly conditional on theta and x_obs."""
    z = np.asarray(log_theta, dtype=float)
    states = simulator._solve_states(np.exp(z)).T.reshape(-1)
    complete = np.asarray(x_obs, dtype=float).copy()
    missing = np.asarray(mask, dtype=bool) == 0
    complete[missing] = rng.lognormal(np.log(np.clip(states[missing], 1e-10, simulator.max_state)), simulator.observation_noise_scale)
    return complete


def importance_reweight_lotka_volterra_mnar(
    log_theta_proposals: np.ndarray, x_obs: np.ndarray, mask: np.ndarray, *, simulator: LotkaVolterraSimulator,
    missing_fraction: float, latent_completions: int, rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Joint IS from p(theta|x_obs) p(z|theta), with exact LV MNAR weights."""
    proposals = np.asarray(log_theta_proposals, dtype=float)
    if proposals.ndim != 2 or proposals.shape[1] != 4 or latent_completions < 1:
        raise ValueError("Expected (n,4) log-theta proposals and positive latent_completions.")
    logw = np.full((len(proposals), latent_completions), -np.inf)
    failures = 0
    for i, theta in enumerate(proposals):
        # One deterministic trajectory per theta; latent completions only vary
        # the missing measurement noise and must not trigger new ODE solves.
        try:
            states = simulator._solve_states(np.exp(theta)).T.reshape(-1)
        except Exception:
            failures += latent_completions
            continue
        for j in range(latent_completions):
            try:
                completed = np.asarray(x_obs, dtype=float).copy()
                missing = np.asarray(mask, dtype=bool) == 0
                completed[missing] = rng.lognormal(np.log(np.clip(states[missing], 1e-10, simulator.max_state)), simulator.observation_noise_scale)
                logw[i, j] = lotka_volterra_log_total_mnar_log_probability(completed, mask, missing_fraction=missing_fraction)
            except Exception:
                failures += 1
    normalizer = logsumexp(logw)
    if not np.isfinite(normalizer): raise FloatingPointError("All LV/MNAR joint importance weights are zero.")
    # Rao--Blackwellize over K completions: each theta gets its average mask
    # probability, not K pseudo-independent parameter weights.
    parameter_logw = logsumexp(logw, axis=1) - np.log(latent_completions)
    parameter_logw -= logsumexp(parameter_logw)
    weights = np.exp(parameter_logw); ess = float(1.0 / np.sum(weights**2))
    return (np.exp(proposals), {"parameter_weights": weights, "importance_ess": ess, "importance_ess_fraction": ess / len(weights),
        "max_normalized_weight": float(weights.max()), "zero_weight_fraction": float(np.mean(~np.isfinite(parameter_logw))),
        "joint_zero_weight_fraction": float(np.mean(~np.isfinite(logw))), "distinct_theta_proposals": int(len(proposals)),
        "latent_completions": int(latent_completions), "ode_or_sampling_failures": failures})


def lotka_volterra_mnar_support_tilt(
    states_flat: np.ndarray, x_obs: np.ndarray, mask: np.ndarray, *, missing_fraction: float,
    sigma: float, safety_margin: float = 0.15,
) -> float:
    """Deterministic log-observation tilt making the central completion feasible."""
    observed = np.asarray(mask).reshape(-1, 2)[:, 0].astype(bool)
    r_obs = np.log(np.asarray(x_obs).reshape(-1, 2)[observed].sum(axis=1) + 1e-12)
    a, b = float(r_obs.min()), float(r_obs.max())
    threshold = a + missing_fraction * (b - a)
    base = np.log(np.asarray(states_flat).reshape(-1, 2).sum(axis=1) + 1e-12)
    missing = ~observed; n = int(missing.sum())
    if n == 0: return 0.0
    lower_min = a - float(base[missing].min())
    lower_mean = (50 * threshold - float(base[observed].sum()) - float(base[missing].sum())) / n
    return float(max(0.0, lower_min, lower_mean) + safety_margin * sigma)


def tilted_lotka_volterra_mnar_importance(
    log_theta_proposals: np.ndarray, x_obs: np.ndarray, mask: np.ndarray, *, simulator: LotkaVolterraSimulator,
    missing_fraction: float, latent_completions: int, rng: np.random.Generator, safety_margin: float = 0.15,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Parameter-level IS with a deterministic shifted-Normal latent proposal."""
    ztheta=np.asarray(log_theta_proposals); n=len(ztheta); logw=np.full((n,latent_completions),-np.inf); lambdas=np.full(n,np.nan); feasible=0; failures=0
    missing=np.asarray(mask,dtype=bool)==0; tm=np.asarray(mask).reshape(-1,2)[:,0].astype(bool); r_obs=np.log(np.asarray(x_obs).reshape(-1,2)[tm].sum(1)+1e-12); a,b=r_obs.min(),r_obs.max(); threshold=a+missing_fraction*(b-a)
    for i,z in enumerate(ztheta):
        try: states=simulator._solve_states(np.exp(z)).T.reshape(-1)
        except Exception: failures+=latent_completions; continue
        lam=lotka_volterra_mnar_support_tilt(states,x_obs,mask,missing_fraction=missing_fraction,sigma=simulator.observation_noise_scale,safety_margin=safety_margin); lambdas[i]=lam; mu=np.log(np.clip(states,1e-10,simulator.max_state))
        for k in range(latent_completions):
            complete=np.asarray(x_obs,float).copy(); y=rng.normal(mu[missing]+lam,simulator.observation_noise_scale);complete[missing]=np.exp(y); rr=np.log(complete.reshape(-1,2).sum(1)+1e-12); ok=np.all(rr[~tm]>a) and rr.mean()>threshold
            feasible+=int(ok)
            if ok:
                correction=float(np.sum(-((y-mu[missing])**2-(y-mu[missing]-lam)**2)/(2*simulator.observation_noise_scale**2)))
                logw[i,k]=lotka_volterra_log_total_mnar_log_probability(complete,mask,missing_fraction=missing_fraction)+correction
    lw=logsumexp(logw,axis=1)-np.log(latent_completions); norm=logsumexp(lw)
    if not np.isfinite(norm): raise FloatingPointError('All support-aware LV/MNAR weights are zero.')
    w=np.exp(lw-norm);return np.exp(ztheta),{'parameter_weights':w,'importance_ess':float(1/(w@w)),'importance_ess_fraction':float(1/(n*(w@w))),'max_normalized_weight':float(w.max()),'zero_weight_fraction':float(np.mean(~np.isfinite(lw))),'feasible_completion_fraction':feasible/(n*latent_completions),'tilt_mean':float(np.nanmean(lambdas)),'tilt_min':float(np.nanmin(lambdas)),'tilt_max':float(np.nanmax(lambdas)),'ode_or_sampling_failures':failures}


def run_lotka_volterra_emcee_reference(
    x_full: np.ndarray,
    *,
    simulator: LotkaVolterraSimulator,
    num_walkers: int,
    burn_in_steps: int,
    production_steps: int,
    initial_scale: float,
    initial_log_theta: np.ndarray | None = None,
    map_log_theta: np.ndarray | None = None,
    map_log_prob: float | None = None,
    walker_center_log_theta: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    screen_initialization: bool = False,
    rng: np.random.Generator,
) -> dict[str, np.ndarray | float | int]:
    """Run one emcee ensemble for an LV observation, sampling z=log(theta)."""
    import emcee

    if (map_log_theta is None) != (map_log_prob is None):
        raise ValueError("map_log_theta and map_log_prob must be supplied together.")
    if map_log_theta is None:
        map_log_theta, map_log_prob = find_lotka_volterra_map(
            x_full,
            simulator,
            initial_log_theta=initial_log_theta,
            mask=mask,
        )
    else:
        map_log_theta = np.asarray(map_log_theta, dtype=float)
        map_log_prob = float(map_log_prob)
        if map_log_theta.shape != (simulator.theta_dim,) or not np.isfinite(map_log_prob):
            raise ValueError("Provided LV MAP must be finite and have simulator.theta_dim entries.")
    walker_center = map_log_theta if walker_center_log_theta is None else np.asarray(walker_center_log_theta, dtype=float)
    if walker_center.shape != (simulator.theta_dim,) or not np.all(np.isfinite(walker_center)):
        raise ValueError("walker_center_log_theta must be a finite vector with simulator.theta_dim entries.")
    if screen_initialization:
        initial_state, initialization_diagnostics = initialize_screened_lotka_volterra_walkers(
            map_log_theta=map_log_theta,
            map_log_prob=map_log_prob,
            simulator=simulator,
            x_full=x_full,
            mask=mask,
            num_walkers=num_walkers,
            fallback_scale=initial_scale,
            rng=rng,
        )
    else:
        initial_state = initialize_lotka_volterra_walkers(
            center=walker_center,
            simulator=simulator,
            num_walkers=num_walkers,
            scale=initial_scale,
            rng=rng,
        )
        initial_log_prob = np.asarray(
            lotka_volterra_log_prob(initial_state, x_full, simulator, mask=mask), dtype=float
        )
        initialization_diagnostics = {
            "method": "unscreened_isotropic_gaussian",
            "initial_log_probabilities": initial_log_prob,
            "initial_log_prob_min_max": np.array([np.min(initial_log_prob), np.max(initial_log_prob)]),
        }
    ode_failure_count = 0
    log_prob_evaluation_count = 0

    def log_prob_fn(log_theta: np.ndarray) -> float:
        nonlocal ode_failure_count, log_prob_evaluation_count
        log_prob_evaluation_count += 1
        value = float(lotka_volterra_log_prob(log_theta, x_full, simulator, mask=mask))
        if not np.isfinite(value):
            ode_failure_count += 1
        return value

    sampler = emcee.EnsembleSampler(
        num_walkers,
        simulator.theta_dim,
        log_prob_fn,
    )
    sampler.run_mcmc(initial_state, burn_in_steps, progress=False)
    sampler.reset()
    ode_failure_count = 0
    sampler.run_mcmc(None, production_steps, progress=False)
    log_theta_chain = sampler.get_chain()
    log_posterior_chain = sampler.get_log_prob()
    chain = np.exp(log_theta_chain)
    acceptance_fraction_by_walker = np.asarray(sampler.acceptance_fraction, dtype=float)
    acceptance_fraction = float(np.mean(acceptance_fraction_by_walker))
    try:
        log_theta_autocorr_time = sampler.get_autocorr_time(tol=0)
        log_theta_emcee_ess = num_walkers * production_steps / log_theta_autocorr_time
    except Exception:
        log_theta_autocorr_time = np.full(simulator.theta_dim, np.nan, dtype=float)
        log_theta_emcee_ess = np.full(simulator.theta_dim, np.nan, dtype=float)
    split_rhat = compute_split_rhat(chain)
    log_theta_split_rhat = compute_split_rhat(log_theta_chain)
    theta_autocorr_time = log_theta_autocorr_time
    theta_emcee_ess = log_theta_emcee_ess
    (
        walker_group_mean_l2_max,
        walker_group_std_l2_max,
    ) = compute_walker_group_agreement(chain)
    return {
        "chain": chain,
        "log_theta_chain": log_theta_chain,
        "log_posterior_chain": log_posterior_chain,
        "acceptance_fraction": acceptance_fraction,
        "acceptance_fraction_by_walker": acceptance_fraction_by_walker,
        "autocorr_time": theta_autocorr_time,
        "emcee_ess": theta_emcee_ess,
        "split_rhat": split_rhat,
        "log_theta_split_rhat": log_theta_split_rhat,
        "log_theta_emcee_ess": log_theta_emcee_ess,
        "walker_group_mean_l2_max": float(walker_group_mean_l2_max),
        "walker_group_std_l2_max": float(walker_group_std_l2_max),
        "map_estimate": np.exp(map_log_theta),
        "map_log_theta": map_log_theta,
        "map_log_prob": float(map_log_prob),
        "walker_center_log_theta": walker_center,
        "initial_state": initial_state,
        "initialization_diagnostics": initialization_diagnostics,
        "mcmc_log_prob_evaluation_count": int(log_prob_evaluation_count),
        "ode_failure_count": int(ode_failure_count),
    }


def find_lotka_volterra_map(
    x_full: np.ndarray,
    simulator: LotkaVolterraSimulator,
    initial_log_theta: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Find an unconstrained LV posterior mode in log-parameter space."""
    starts = [
        simulator.prior_log_mean,
        simulator.prior_log_mean + np.array([0.35, 0.0, 0.35, 0.0]),
        simulator.prior_log_mean + np.array([-0.35, 0.0, -0.35, 0.0]),
        simulator.prior_log_mean + np.array([0.0, 0.35, 0.0, 0.35]),
        simulator.prior_log_mean + np.array([0.0, -0.35, 0.0, -0.35]),
    ]
    if initial_log_theta is not None:
        starts.insert(0, np.asarray(initial_log_theta, dtype=float))

    def objective(log_theta: np.ndarray) -> float:
        value = float(lotka_volterra_log_prob(log_theta, x_full, simulator, mask=mask))
        if not np.isfinite(value):
            return 1e100
        return -value

    best_x = np.asarray(starts[0], dtype=float)
    best_fun = objective(best_x)
    for start in starts:
        result = minimize(
            objective,
            np.asarray(start, dtype=float),
            method="Nelder-Mead",
            options={"maxiter": 1200, "xatol": 1e-4, "fatol": 1e-4},
        )
        if result.success and float(result.fun) < best_fun:
            best_x = np.asarray(result.x, dtype=float)
            best_fun = float(result.fun)
    return best_x, -best_fun


def initialize_lotka_volterra_walkers(
    *,
    center: np.ndarray,
    simulator: LotkaVolterraSimulator,
    num_walkers: int,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Initialize LV log-theta walkers near a mode with finite prior density."""
    walkers = rng.normal(
        loc=center,
        scale=scale,
        size=(num_walkers, simulator.theta_dim),
    )
    invalid = ~np.all(np.isfinite(walkers), axis=1)
    while np.any(invalid):
        walkers[invalid] = rng.normal(
            loc=center,
            scale=scale,
            size=(int(np.sum(invalid)), simulator.theta_dim),
        )
        invalid = ~np.all(np.isfinite(walkers), axis=1)
    return walkers


def _finite_difference_negative_log_hessian(
    log_probability: Any, center: np.ndarray
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Numerically estimate the local negative-log-posterior Hessian."""
    center = np.asarray(center, dtype=float)
    dimension = center.size
    steps = 1e-3 * np.maximum(1.0, np.abs(center))
    evaluation_count = 1
    f0 = float(log_probability(center))
    if not np.isfinite(f0):
        return None, {"reason": "nonfinite_map_log_prob", "log_probability_evaluation_count": evaluation_count}
    hessian = np.empty((dimension, dimension), dtype=float)
    try:
        for i in range(dimension):
            displacement = np.zeros(dimension); displacement[i] = steps[i]
            plus, minus = float(log_probability(center + displacement)), float(log_probability(center - displacement))
            evaluation_count += 2
            if not np.isfinite(plus) or not np.isfinite(minus):
                return None, {"reason": "nonfinite_diagonal_difference", "log_probability_evaluation_count": evaluation_count}
            hessian[i, i] = -(plus - 2.0 * f0 + minus) / steps[i] ** 2
            for j in range(i):
                displacement_j = np.zeros(dimension); displacement_j[j] = steps[j]
                values = [
                    float(log_probability(center + displacement + displacement_j)),
                    float(log_probability(center + displacement - displacement_j)),
                    float(log_probability(center - displacement + displacement_j)),
                    float(log_probability(center - displacement - displacement_j)),
                ]
                evaluation_count += 4
                if not np.all(np.isfinite(values)):
                    return None, {"reason": "nonfinite_cross_difference", "log_probability_evaluation_count": evaluation_count}
                value = -(values[0] - values[1] - values[2] + values[3]) / (4.0 * steps[i] * steps[j])
                hessian[i, j] = hessian[j, i] = value
    except Exception as error:
        return None, {"reason": type(error).__name__, "log_probability_evaluation_count": evaluation_count}
    hessian = (hessian + hessian.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(hessian)
    condition_number = float(np.max(eigenvalues) / np.min(eigenvalues)) if np.min(eigenvalues) > 0 else np.inf
    diagnostic = {"steps": steps, "hessian": hessian, "eigenvalues": eigenvalues, "condition_number": condition_number,
                  "log_probability_evaluation_count": evaluation_count}
    if not np.all(np.isfinite(eigenvalues)) or np.min(eigenvalues) <= 0 or condition_number > 1e8:
        diagnostic["reason"] = "not_positive_definite_or_ill_conditioned"
        return None, diagnostic
    return hessian, diagnostic


def initialize_screened_lotka_volterra_walkers(
    *,
    map_log_theta: np.ndarray,
    map_log_prob: float,
    simulator: LotkaVolterraSimulator,
    x_full: np.ndarray,
    mask: np.ndarray | None,
    num_walkers: int,
    fallback_scale: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Draw a dispersed, finite local LV ensemble before MCMC starts.

    The screen is a *proposal-initialization* check, not a target constraint.
    Its drop is half the 1-1e-6 chi-square radius in four dimensions: under a
    local Gaussian approximation fewer than one in a million valid local
    draws would be rejected solely by this threshold.
    """
    if num_walkers < 2 * simulator.theta_dim:
        raise ValueError("num_walkers must be at least twice the LV parameter dimension.")
    if not np.isfinite(map_log_prob):
        raise ValueError("A finite partial-data MAP is required for screened initialization.")
    threshold_drop = float(0.5 * chi2.ppf(1.0 - 1e-6, simulator.theta_dim))

    def log_probability(values: np.ndarray) -> np.ndarray | float:
        return lotka_volterra_log_prob(values, x_full, simulator, mask=mask)

    hessian, hessian_diagnostics = _finite_difference_negative_log_hessian(log_probability, map_log_theta)
    if hessian is None:
        covariance = np.eye(simulator.theta_dim) * fallback_scale**2
        method = "screened_isotropic_fallback"
    else:
        covariance = np.linalg.inv(hessian)
        method = "screened_inverse_negative_log_hessian"
    covariance = (covariance + covariance.T) / 2.0
    scale_factor = 1.0
    accepted: list[np.ndarray] = []
    candidate_count = rejection_count = 0
    maximum_batches = 100
    for batch in range(maximum_batches):
        remaining = num_walkers - len(accepted)
        if remaining == 0:
            break
        candidates = rng.multivariate_normal(map_log_theta, covariance * scale_factor**2, size=max(remaining * 4, 16))
        values = np.asarray(log_probability(candidates), dtype=float)
        candidate_count += candidates.shape[0]
        keep = np.isfinite(values) & (values >= map_log_prob - threshold_drop)
        rejection_count += int(np.count_nonzero(~keep))
        accepted.extend(candidates[keep][:remaining])
        # A poor Hessian approximation can be too broad far from the mode;
        # adapt only the *initializer* until it reaches the justified contour.
        if not np.any(keep):
            scale_factor *= 0.5
    if len(accepted) != num_walkers:
        raise RuntimeError("Could not construct a fully screened LV walker ensemble.")
    walkers = np.asarray(accepted, dtype=float)
    initial_log_probabilities = np.asarray(log_probability(walkers), dtype=float)
    centered = walkers - walkers.mean(axis=0)
    if np.linalg.matrix_rank(centered) < simulator.theta_dim:
        raise RuntimeError("Screened LV walker ensemble is degenerate.")
    return walkers, {
        "method": method,
        "screen_log_density_drop": threshold_drop,
        "screen_probability_coverage": 1.0 - 1e-6,
        "map_log_prob": float(map_log_prob),
        "initial_log_probabilities": initial_log_probabilities,
        "initial_log_prob_min_max": np.array([initial_log_probabilities.min(), initial_log_probabilities.max()]),
        "proposal_covariance": covariance,
        "proposal_scale_factor": scale_factor,
        "candidate_count": candidate_count,
        "rejection_count": rejection_count,
        "hessian_diagnostics": hessian_diagnostics,
    }


def lotka_volterra_posterior_predictive_log_rmse(
    theta_samples: np.ndarray,
    x_full: np.ndarray,
    *,
    simulator: LotkaVolterraSimulator,
    max_samples: int = 1_000,
) -> np.ndarray:
    """Compute deterministic posterior-predictive log RMSE for LV theta samples."""
    samples = np.asarray(theta_samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != simulator.theta_dim:
        raise ValueError("theta_samples must have shape (num_samples, 4).")
    if samples.shape[0] > max_samples:
        indices = np.linspace(0, samples.shape[0] - 1, max_samples, dtype=int)
        samples = samples[indices]
    observed_log = np.log(np.asarray(x_full, dtype=float))
    rmse = np.full(samples.shape[0], np.nan, dtype=float)
    for idx, theta in enumerate(samples):
        try:
            states = simulator._solve_states(theta)
        except Exception:
            continue
        predicted_log = np.log(np.clip(states.T.reshape(-1), 1e-10, simulator.max_state))
        rmse[idx] = float(np.sqrt(np.mean((predicted_log - observed_log) ** 2)))
    return rmse[np.isfinite(rmse)]


def compute_split_rhat(chain: np.ndarray) -> np.ndarray:
    """Compute split R-hat treating walkers as chains."""
    chain_array = np.asarray(chain, dtype=float)
    n_steps, n_walkers, n_dim = chain_array.shape
    half = n_steps // 2
    if half < 2:
        return np.full(n_dim, np.nan, dtype=float)
    split = np.concatenate(
        [chain_array[:half], chain_array[-half:]],
        axis=1,
    )
    chains = np.transpose(split, (1, 0, 2))
    chain_means = chains.mean(axis=1)
    chain_vars = chains.var(axis=1, ddof=1)
    within = chain_vars.mean(axis=0)
    between = half * chain_means.var(axis=0, ddof=1)
    var_hat = ((half - 1) / half) * within + between / half
    return np.sqrt(var_hat / within)


def compute_walker_group_agreement(
    chain: np.ndarray,
    *,
    num_groups: int = 4,
) -> tuple[float, float]:
    """Compare posterior moments across walker groups."""
    chain_array = np.asarray(chain, dtype=float)
    n_walkers = chain_array.shape[1]
    groups = np.array_split(np.arange(n_walkers), num_groups)
    flat = chain_array.reshape(-1, chain_array.shape[-1])
    overall_mean = flat.mean(axis=0)
    overall_std = flat.std(axis=0)
    mean_deltas = []
    std_deltas = []
    for group in groups:
        group_flat = chain_array[:, group, :].reshape(-1, chain_array.shape[-1])
        mean_deltas.append(np.linalg.norm(group_flat.mean(axis=0) - overall_mean))
        std_deltas.append(np.linalg.norm(group_flat.std(axis=0) - overall_std))
    return float(np.max(mean_deltas)), float(np.max(std_deltas))


def oup_log_likelihood(
    theta: np.ndarray,
    x_full: np.ndarray,
    simulator: OUPSimulator | None = None,
) -> np.ndarray:
    """Evaluate the OUP log likelihood matching the repository simulator."""
    simulator = OUPSimulator() if simulator is None else simulator
    theta_array = np.asarray(theta, dtype=float)
    single = theta_array.ndim == 1
    if single:
        theta_array = theta_array[None, :]
    if theta_array.ndim != 2 or theta_array.shape[1] != simulator.theta_dim:
        raise ValueError("theta must have shape (2,) or (batch, 2).")
    x = np.asarray(x_full, dtype=float)
    if x.shape != simulator.x_shape:
        raise ValueError(f"x_full must have shape {simulator.x_shape}.")
    if simulator.var <= 0:
        raise ValueError("OUP likelihood requires positive simulator variance.")

    log_like = np.full(theta_array.shape[0], -np.inf, dtype=float)
    valid = (
        (theta_array[:, 0] >= simulator.prior_low[0])
        & (theta_array[:, 0] <= simulator.prior_high[0])
        & (theta_array[:, 1] >= simulator.prior_low[1])
        & (theta_array[:, 1] <= simulator.prior_high[1])
    )
    if not np.any(valid):
        return log_like[0] if single else log_like
    if not np.isclose(x[0], simulator.y0, rtol=0.0, atol=1e-10):
        return log_like[0] if single else log_like

    theta_valid = theta_array[valid]
    theta1 = theta_valid[:, 0]
    theta2_exp = np.exp(theta_valid[:, 1])
    y_t = x[:-1]
    y_next = x[1:]
    mean = y_t[None, :] + theta1[:, None] * (theta2_exp[:, None] - y_t[None, :]) * simulator.dt
    noise_var = 0.25 * simulator.dt * simulator.var
    residual = y_next[None, :] - mean
    log_norm = np.log(2.0 * np.pi * noise_var)
    log_like[valid] = -0.5 * np.sum((residual * residual) / noise_var + log_norm, axis=1)
    return log_like[0] if single else log_like


def oup_mcar_observed_log_likelihood(
    theta: np.ndarray,
    x_obs: np.ndarray,
    mask: np.ndarray,
    simulator: OUPSimulator | None = None,
) -> np.ndarray:
    """OUP likelihood after exactly marginalizing MCAR-masked Euler states.

    ``mask == 1`` denotes an observed state.  The deterministic simulator
    initial state is always retained as the transition anchor, even if its
    corresponding mask entry is zero.  This is a likelihood only: the MCAR
    mask law is parameter independent and is therefore intentionally omitted.
    """
    simulator = OUPSimulator() if simulator is None else simulator
    theta_array = np.asarray(theta, dtype=float)
    single = theta_array.ndim == 1
    if single:
        theta_array = theta_array[None, :]
    if theta_array.ndim != 2 or theta_array.shape[1] != simulator.theta_dim:
        raise ValueError("theta must have shape (2,) or (batch, 2).")
    x = np.asarray(x_obs, dtype=float)
    observed = np.asarray(mask)
    if x.shape != simulator.x_shape or observed.shape != simulator.x_shape:
        raise ValueError("x_obs and mask must have shape matching simulator.x_shape.")
    if not np.all((observed == 0) | (observed == 1)):
        raise ValueError("mask must be binary with 1=observed and 0=missing.")
    if simulator.var <= 0:
        raise ValueError("OUP likelihood requires positive simulator variance.")

    # Preserve the established full-data calculation bit-for-bit where possible.
    if np.all(observed == 1):
        return oup_log_likelihood(theta_array, x, simulator)[0] if single else oup_log_likelihood(theta_array, x, simulator)

    log_like = np.full(theta_array.shape[0], -np.inf, dtype=float)
    valid = (
        (theta_array[:, 0] >= simulator.prior_low[0])
        & (theta_array[:, 0] <= simulator.prior_high[0])
        & (theta_array[:, 1] >= simulator.prior_low[1])
        & (theta_array[:, 1] <= simulator.prior_high[1])
    )
    # A reported initial observation must equal the known deterministic state.
    if observed[0] and not np.isclose(x[0], simulator.y0, rtol=0.0, atol=1e-10):
        return log_like[0] if single else log_like
    if not np.any(valid):
        return log_like[0] if single else log_like

    theta_valid = theta_array[valid]
    kappa = theta_valid[:, 0]
    mu = np.exp(theta_valid[:, 1])
    a = 1.0 - kappa * simulator.dt
    b = kappa * mu * simulator.dt
    q = 0.25 * simulator.dt * simulator.var

    current_index = 0
    current_value = float(simulator.y0)
    accumulated = np.zeros(theta_valid.shape[0], dtype=float)
    for next_index in np.flatnonzero(observed[1:]) + 1:
        gap = int(next_index - current_index)
        # Finite sums are stable at kappa=0 (a=1), and gaps are at most n-1.
        powers = np.arange(gap, dtype=float)
        sum_a = np.sum(a[:, None] ** powers[None, :], axis=1)
        sum_a2 = np.sum((a * a)[:, None] ** powers[None, :], axis=1)
        mean = (a**gap) * current_value + b * sum_a
        variance = q * sum_a2
        residual = x[next_index] - mean
        accumulated += -0.5 * (
            residual * residual / variance + np.log(2.0 * np.pi * variance)
        )
        current_index = int(next_index)
        current_value = float(x[next_index])

    log_like[valid] = accumulated
    return log_like[0] if single else log_like


def oup_mnar_mask_log_probability(
    x_full: np.ndarray,
    mask: np.ndarray,
    *,
    missing_fraction: float,
    eps: float = 1e-12,
) -> float | np.ndarray:
    """Exact log mask probability for identity mean-normalized self-censoring.

    This is a vectorized transcription of
    :class:`MeanNormalizedSelfCensoringMNARMask`, including its constant-row
    fallback and hard probabilities caused by clipping.  A masked fixed OUP
    initial state remains part of ``x_full`` and therefore of the score.
    """
    if not 0.0 <= missing_fraction <= 1.0 or eps <= 0.0:
        raise ValueError("missing_fraction must be in [0, 1] and eps must be positive.")
    x = np.asarray(x_full, dtype=float)
    m = np.asarray(mask, dtype=np.int8)
    single = x.ndim == 1
    if single:
        x = x[None, :]
    if x.ndim != 2 or m.ndim != 1 or m.shape[0] != x.shape[1]:
        raise ValueError("x_full must be (time,) or (batch, time), with a matching one-dimensional mask.")
    if not np.all(np.isfinite(x)) or not np.all((m == 0) | (m == 1)):
        raise ValueError("x_full must be finite and mask must be binary.")
    minimum = x.min(axis=1, keepdims=True)
    maximum = x.max(axis=1, keepdims=True)
    value_range = maximum - minimum
    raw_score = np.divide(
        x - minimum, value_range + eps, out=np.zeros_like(x), where=value_range >= eps
    )
    raw_score = np.where(value_range < eps, 0.5, raw_score)
    score = raw_score / (raw_score.mean(axis=1, keepdims=True) + eps)
    p_missing = np.clip(missing_fraction * score, 0.0, 1.0)
    impossible = np.any(((m == 0) & (p_missing == 0.0)) | ((m == 1) & (p_missing == 1.0)), axis=1)
    with np.errstate(divide="ignore"):
        log_terms = np.where(m == 0, np.log(p_missing), np.log1p(-p_missing))
    result = np.where(impossible, -np.inf, log_terms.sum(axis=1))
    return float(result[0]) if single else result


def sample_oup_conditional_gaussian_trajectories(
    theta: np.ndarray,
    x_obs: np.ndarray,
    mask: np.ndarray,
    *,
    simulator: OUPSimulator | None = None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw exact Euler-chain Gaussian bridges conditional on observed states.

    The fixed ``Y_0`` is always an anchor, whether or not it was reported by
    the mask.  Rows of ``theta`` are independent proposals and are sampled in
    parallel; no unobserved value from ``x_obs`` is read.
    """
    simulator = OUPSimulator() if simulator is None else simulator
    theta_array = np.asarray(theta, dtype=float)
    single = theta_array.ndim == 1
    if single:
        theta_array = theta_array[None, :]
    x = np.asarray(x_obs, dtype=float)
    observed = np.asarray(mask, dtype=np.int8)
    if theta_array.ndim != 2 or theta_array.shape[1] != 2:
        raise ValueError("theta must have shape (2,) or (batch, 2).")
    if x.shape != simulator.x_shape or observed.shape != simulator.x_shape:
        raise ValueError("x_obs and mask must match simulator.x_shape.")
    if not np.all((observed == 0) | (observed == 1)):
        raise ValueError("mask must be binary.")
    if observed[0] and not np.isclose(x[0], simulator.y0, rtol=0.0, atol=1e-10):
        raise ValueError("An observed OUP initial state must equal the deterministic y0.")
    if np.any(theta_array[:, 0] < simulator.prior_low[0]) or np.any(theta_array[:, 0] > simulator.prior_high[0]) or np.any(theta_array[:, 1] < simulator.prior_low[1]) or np.any(theta_array[:, 1] > simulator.prior_high[1]):
        raise ValueError("theta lies outside the OUP prior support.")

    batch = theta_array.shape[0]
    a = 1.0 - theta_array[:, 0] * simulator.dt
    b = theta_array[:, 0] * np.exp(theta_array[:, 1]) * simulator.dt
    q = 0.25 * simulator.dt * simulator.var
    completed = np.empty((batch, simulator.n), dtype=float)
    completed[:, 0] = simulator.y0
    anchors = np.unique(np.concatenate(([0], np.flatnonzero(observed[1:]) + 1)))
    for left, right in zip(anchors[:-1], anchors[1:], strict=True):
        current = completed[:, left]
        gap = int(right - left)
        endpoint = float(x[right])
        for step in range(1, gap):
            remaining = gap - step
            prior_mean = a * current + b
            end_powers = np.arange(remaining + 1, dtype=float)
            end_mean = (a ** (remaining + 1)) * current + b * np.sum(a[:, None] ** end_powers[None, :], axis=1)
            end_variance = q * np.sum((a * a)[:, None] ** end_powers[None, :], axis=1)
            covariance = q * (a ** remaining)
            gain = covariance / end_variance
            conditional_mean = prior_mean + gain * (endpoint - end_mean)
            conditional_variance = q - covariance * gain
            completed[:, left + step] = conditional_mean + np.sqrt(np.maximum(conditional_variance, 0.0)) * rng.normal(size=batch)
            current = completed[:, left + step]
        completed[:, right] = endpoint
    last = int(anchors[-1])
    current = completed[:, last]
    for index in range(last + 1, simulator.n):
        current = a * current + b + np.sqrt(q) * rng.normal(size=batch)
        completed[:, index] = current
    return completed[0] if single else completed


def _weighted_oup_theta_summary(theta: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    mean = weights @ theta
    centered = theta - mean
    covariance = centered.T @ (centered * weights[:, None])
    quantiles = np.empty((theta.shape[1], 3), dtype=float)
    for dimension in range(theta.shape[1]):
        order = np.argsort(theta[:, dimension])
        cdf = np.cumsum(weights[order])
        quantiles[dimension] = theta[order[np.searchsorted(cdf, (0.05, 0.5, 0.95))], dimension]
    return {"mean": mean, "covariance": covariance, "quantiles": quantiles}


def sample_oup_mnar_importance(
    *,
    x_obs: np.ndarray,
    mask: np.ndarray,
    missing_fraction: float,
    mcar_posterior: dict[str, np.ndarray | float | int],
    num_proposals: int,
    num_samples: int,
    rng: np.random.Generator,
    simulator: OUPSimulator | None = None,
    batch_size: int = 10_000,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Joint IS from the exact MCAR posterior and Gaussian bridge proposal.

    The proposal is ``p(theta | x_obs) p(x_mis | theta, x_obs)``.  Hence the
    only importance factor is the exact MNAR mask likelihood.
    """
    if num_proposals <= 0 or num_samples <= 0 or batch_size <= 0:
        raise ValueError("num_proposals, num_samples, and batch_size must be positive.")
    simulator = OUPSimulator() if simulator is None else simulator
    theta_proposals = np.empty((num_proposals, simulator.theta_dim), dtype=float)
    log_weights = np.empty(num_proposals, dtype=float)
    started = time.perf_counter()
    for start in range(0, num_proposals, batch_size):
        stop = min(start + batch_size, num_proposals)
        theta_batch = sample_oup_grid_posterior(
            mcar_posterior, num_reference_samples=stop - start, rng=rng
        )
        completed = sample_oup_conditional_gaussian_trajectories(
            theta_batch, x_obs, mask, simulator=simulator, rng=rng
        )
        theta_proposals[start:stop] = theta_batch
        log_weights[start:stop] = oup_mnar_mask_log_probability(
            completed, mask, missing_fraction=missing_fraction
        )
    normalizer = float(logsumexp(log_weights))
    if not np.isfinite(normalizer):
        raise FloatingPointError("All OUP/MNAR importance proposals have zero weight.")
    weights = np.exp(log_weights - normalizer)
    ess = float(1.0 / np.sum(weights * weights))
    # Systematic resampling is lower variance than independent categorical draws.
    positions = (rng.uniform() + np.arange(num_samples)) / num_samples
    indices = np.searchsorted(np.cumsum(weights), positions, side="right")
    indices = np.minimum(indices, num_proposals - 1)
    summary = _weighted_oup_theta_summary(theta_proposals, weights)
    diagnostics: dict[str, Any] = {
        "num_proposals": int(num_proposals), "num_samples": int(num_samples),
        "batch_size": int(batch_size), "ess": ess,
        "ess_fraction": ess / num_proposals,
        "max_normalized_weight": float(weights.max()),
        "zero_weight_fraction": float(np.mean(~np.isfinite(log_weights))),
        "num_distinct_resampled_proposals": int(np.unique(indices).size),
        "runtime_seconds": time.perf_counter() - started,
        "posterior_mean": summary["mean"], "posterior_covariance": summary["covariance"],
        "posterior_quantiles": summary["quantiles"],
        "log_weight_normalizer": normalizer,
    }
    return theta_proposals[indices], diagnostics


def generate_oup_mnar_partial_data_reference_posteriors(
    *,
    full_reference_path: str | os.PathLike[str] = "references/reference_posteriors_v1/oup_references.h5",
    missing_fraction: float,
    reference_mask_seed: int = 987_654,
    posterior_seed: int = 23_456,
    num_proposals: int = 1_000_000,
    num_reference_samples: int = 10_000,
    grid_resolution: int = 4_800,
    chunk_size: int = 200_000,
    min_ess: float = 20_000.0,
    min_ess_fraction: float = 0.02,
    max_normalized_weight: float = 0.001,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, np.ndarray]]:
    """Generate validated OUP/MNAR references by joint bridge importance sampling."""
    if min_ess <= 0 or not 0 < min_ess_fraction <= 1 or not 0 < max_normalized_weight <= 1:
        raise ValueError("Invalid importance-sampling quality gates.")
    full_obs, _, full_meta, _ = load_reference_posteriors_hdf5(full_reference_path)
    if full_meta.get("problem") != "oup":
        raise ValueError("full_reference_path must be an OUP reference artifact.")
    sm = full_meta["simulator"]
    simulator = OUPSimulator(n=int(sm["n"]), T=float(sm["T"]), var=float(sm["var"]), y0=float(sm["y0"]))
    theta_true = np.asarray(full_obs["theta_true"], float)
    x_full = np.asarray(full_obs["x_full"], float)
    generator = MeanNormalizedSelfCensoringMNARMask(
        missing_fraction=missing_fraction, eps=1e-12, score_transform="identity"
    )
    mask = generator.generate(x_full, theta_true, np.random.default_rng(reference_mask_seed))
    x_obs = x_full * mask
    rng = np.random.default_rng(posterior_seed)
    n = x_full.shape[0]
    theta_samples = np.empty((n, num_reference_samples, 2), float)
    ess = np.empty(n); ess_fraction = np.empty(n); max_weight = np.empty(n)
    zero_weight_fraction = np.empty(n); distinct = np.empty(n, dtype=np.int64)
    runtime = np.empty(n); grid_runtime = np.empty(n); grid_boundary = np.empty(n)
    mean = np.empty((n, 2)); covariance = np.empty((n, 2, 2)); quantiles = np.empty((n, 2, 3))
    # Mask-only complexity selection; never use the hidden trajectory values.
    def complexity(row: np.ndarray) -> tuple[int, int, int]:
        anchors = np.unique(np.concatenate(([0], np.flatnonzero(row[1:]) + 1)))
        return (int(np.count_nonzero(row == 0)), int(np.diff(anchors).max(initial=0)), int(row[0] == 0))
    validation_indices = np.array([max(range(n), key=lambda j: complexity(mask[j]))], dtype=np.int64)
    validation_samples = np.empty((1, num_reference_samples, 2), float)
    validation_mean_delta = np.empty((1, 2)); validation_cov_delta = np.empty(1)
    validation_quantile_delta = np.empty((1, 2, 3)); validation_ess = np.empty(1)
    validation_seed = np.empty(1, dtype=np.int64)
    import resource
    peak_rss = np.empty(n)
    for index in range(n):
        started = time.perf_counter(); grid_started = started
        mcar = compute_oup_grid_posterior(x_obs[index], simulator=simulator, grid_resolution=grid_resolution, chunk_size=chunk_size, mask=mask[index])
        grid_runtime[index] = time.perf_counter() - grid_started
        grid_boundary[index] = float(mcar["boundary_mass"])
        samples, details = sample_oup_mnar_importance(
            x_obs=x_obs[index], mask=mask[index], missing_fraction=missing_fraction,
            mcar_posterior=mcar, num_proposals=num_proposals, num_samples=num_reference_samples,
            rng=np.random.default_rng(rng.integers(0, np.iinfo(np.uint32).max)), simulator=simulator,
        )
        if (details["ess"] < min_ess or details["ess_fraction"] < min_ess_fraction
                or details["max_normalized_weight"] > max_normalized_weight):
            raise RuntimeError(f"OUP/MNAR observation {index} failed IS quality gates: {details}")
        theta_samples[index] = samples
        ess[index], ess_fraction[index], max_weight[index] = details["ess"], details["ess_fraction"], details["max_normalized_weight"]
        zero_weight_fraction[index], distinct[index] = details["zero_weight_fraction"], details["num_distinct_resampled_proposals"]
        mean[index], covariance[index], quantiles[index] = details["posterior_mean"], details["posterior_covariance"], details["posterior_quantiles"]
        runtime[index] = time.perf_counter() - started
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss[index] = rss / (1024**2) if os.uname().sysname == "Darwin" else rss / 1024
        if index in validation_indices:
            local = int(np.where(validation_indices == index)[0][0]); seed = int(rng.integers(0, np.iinfo(np.uint32).max))
            _, check = sample_oup_mnar_importance(x_obs=x_obs[index], mask=mask[index], missing_fraction=missing_fraction, mcar_posterior=mcar, num_proposals=num_proposals, num_samples=num_reference_samples, rng=np.random.default_rng(seed), simulator=simulator)
            validation_seed[local], validation_ess[local] = seed, check["ess"]
            validation_mean_delta[local] = check["posterior_mean"] - mean[index]
            validation_cov_delta[local] = np.linalg.norm(check["posterior_covariance"] - covariance[index], ord="fro")
            validation_quantile_delta[local] = check["posterior_quantiles"] - quantiles[index]
    observations = {"theta_true": theta_true, "x_full": x_full, "x_obs": x_obs, "mask": mask}
    metadata = {"problem":"oup", "missingness_mechanism":"mnar", "posterior_method":"mcar_grid_gaussian_bridge_joint_importance_sampling", "posterior_conditioning":"theta_given_x_obs_mask", "mask":generator.metadata(), "reference_mask_seed":int(reference_mask_seed), "posterior_seed":int(posterior_seed), "full_data_reference_path":str(full_reference_path), "full_data_reference_version":full_meta.get("reference_version"), "simulator":simulator.metadata(), "theta_scaled":False, "x_scaled":False, "grid_resolution":int(grid_resolution), "num_proposals":int(num_proposals), "resampling":"systematic", "peak_rss_unit":"MiB"}
    diagnostics = {"posterior_sample_mean":theta_samples.mean(1), "posterior_sample_std":theta_samples.std(1), "importance_ess":ess, "importance_ess_fraction":ess_fraction, "importance_max_normalized_weight":max_weight, "importance_zero_weight_fraction":zero_weight_fraction, "importance_num_distinct_resampled_proposals":distinct, "proposal_count":np.full(n,num_proposals,dtype=np.int64), "posterior_mean_weighted":mean, "posterior_covariance_weighted":covariance, "posterior_quantiles_weighted":quantiles, "runtime_seconds":runtime, "grid_runtime_seconds":grid_runtime, "peak_rss_mb":peak_rss, "grid_boundary_mass":grid_boundary, "validation_observation_indices":validation_indices, "validation_seed":validation_seed, "validation_ess":validation_ess, "validation_mean_delta":validation_mean_delta, "validation_cov_fro_delta":validation_cov_delta, "validation_quantile_delta":validation_quantile_delta, "num_nonfinite_posterior_samples":np.array(np.count_nonzero(~np.isfinite(theta_samples))), "num_prior_bound_violations":np.array(np.count_nonzero((theta_samples<simulator.prior_low)|(theta_samples>simulator.prior_high)))}
    return observations, metadata, {"theta_samples":theta_samples, **diagnostics}


def compute_oup_grid_posterior(
    x_full: np.ndarray,
    *,
    simulator: OUPSimulator | None = None,
    grid_resolution: int,
    chunk_size: int = 200_000,
    mask: np.ndarray | None = None,
) -> dict[str, np.ndarray | float | int]:
    """Approximate a full- or MCAR-partial OUP posterior over the prior grid.

    Passing ``mask`` switches only the likelihood to the exact observed-data
    likelihood; all grid normalization and support handling are unchanged.
    """
    simulator = OUPSimulator() if simulator is None else simulator
    if grid_resolution <= 1:
        raise ValueError("grid_resolution must be greater than 1.")
    theta1_step = (simulator.prior_high[0] - simulator.prior_low[0]) / grid_resolution
    theta2_step = (simulator.prior_high[1] - simulator.prior_low[1]) / grid_resolution
    theta1 = simulator.prior_low[0] + (np.arange(grid_resolution) + 0.5) * theta1_step
    theta2 = simulator.prior_low[1] + (np.arange(grid_resolution) + 0.5) * theta2_step
    theta1_grid, theta2_grid = np.meshgrid(theta1, theta2, indexing="ij")
    grid_theta = np.column_stack([theta1_grid.ravel(), theta2_grid.ravel()])

    log_weights = np.empty(grid_theta.shape[0], dtype=float)
    for start in range(0, grid_theta.shape[0], chunk_size):
        stop = min(start + chunk_size, grid_theta.shape[0])
        if mask is None:
            log_weights[start:stop] = oup_log_likelihood(
                grid_theta[start:stop], x_full, simulator
            )
        else:
            log_weights[start:stop] = oup_mcar_observed_log_likelihood(
                grid_theta[start:stop], x_full, mask, simulator
            )
    log_total = float(logsumexp(log_weights))
    if not np.isfinite(log_total):
        raise FloatingPointError("OUP grid posterior has no finite mass.")
    weights = np.exp(log_weights - log_total)
    weights_2d = weights.reshape(grid_resolution, grid_resolution)
    mean = weights @ grid_theta
    centered = grid_theta - mean
    cov = centered.T @ (centered * weights[:, None])
    ess = float(1.0 / np.sum(weights * weights))
    boundary_mask = np.zeros((grid_resolution, grid_resolution), dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True
    boundary_mass = float(np.sum(weights_2d[boundary_mask]))
    return {
        "grid_resolution": int(grid_resolution),
        "grid_theta": grid_theta,
        "weights": weights,
        "mean": mean,
        "cov": cov,
        "ess": ess,
        "boundary_mass": boundary_mass,
        "log_normalizer": log_total + np.log(theta1_step * theta2_step),
        "cell_width": np.array([theta1_step, theta2_step], dtype=float),
    }


def sample_oup_grid_posterior(
    posterior: dict[str, np.ndarray | float | int],
    *,
    num_reference_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw jittered samples from a weighted OUP posterior grid."""
    if num_reference_samples <= 0:
        raise ValueError("num_reference_samples must be positive.")
    grid_theta = np.asarray(posterior["grid_theta"], dtype=float)
    weights = np.asarray(posterior["weights"], dtype=float)
    cell_width = np.asarray(posterior["cell_width"], dtype=float)
    indices = rng.choice(grid_theta.shape[0], size=num_reference_samples, p=weights)
    jitter = rng.uniform(-0.5, 0.5, size=(num_reference_samples, 2)) * cell_width
    samples = grid_theta[indices] + jitter
    return samples


def compute_oup_grid_resolution_diagnostics(
    x_full: np.ndarray,
    *,
    simulator: OUPSimulator,
    reference_posteriors: list[dict[str, np.ndarray | float | int]],
    validation_grid_resolutions: tuple[int, ...],
    observation_indices: tuple[int, ...] | None = None,
    prefix: str = "validation",
) -> dict[str, np.ndarray]:
    """Compare OUP grid moments against the final reference grid."""
    x = np.asarray(x_full, dtype=float)
    if observation_indices is None:
        obs_indices = tuple(range(x.shape[0]))
    else:
        obs_indices = tuple(int(idx) for idx in observation_indices)
    resolutions = tuple(int(resolution) for resolution in validation_grid_resolutions)
    mean_delta = np.full((len(resolutions), len(obs_indices)), np.nan, dtype=float)
    cov_delta = np.full((len(resolutions), len(obs_indices)), np.nan, dtype=float)
    ess = np.full((len(resolutions), len(obs_indices)), np.nan, dtype=float)
    boundary_mass = np.full((len(resolutions), len(obs_indices)), np.nan, dtype=float)
    for res_idx, resolution in enumerate(resolutions):
        for local_obs_idx, obs_idx in enumerate(obs_indices):
            posterior = compute_oup_grid_posterior(
                x[obs_idx],
                simulator=simulator,
                grid_resolution=resolution,
            )
            ref = reference_posteriors[obs_idx]
            mean_delta[res_idx, local_obs_idx] = float(
                np.linalg.norm(np.asarray(posterior["mean"]) - np.asarray(ref["mean"]))
            )
            cov_delta[res_idx, local_obs_idx] = float(
                np.linalg.norm(np.asarray(posterior["cov"]) - np.asarray(ref["cov"]), ord="fro")
            )
            ess[res_idx, local_obs_idx] = float(posterior["ess"])
            boundary_mass[res_idx, local_obs_idx] = float(posterior["boundary_mass"])
    return {
        f"{prefix}_grid_resolutions": np.asarray(resolutions, dtype=np.int64),
        f"{prefix}_observation_indices": np.asarray(obs_indices, dtype=np.int64),
        f"{prefix}_mean_l2_delta_vs_final_grid": mean_delta,
        f"{prefix}_cov_fro_delta_vs_final_grid": cov_delta,
        f"{prefix}_grid_ess": ess,
        f"{prefix}_grid_boundary_mass": boundary_mass,
    }


def _oup_grid_marginal_credible_intervals(
    posterior: dict[str, np.ndarray | float | int],
    probabilities: tuple[float, ...] = (0.025, 0.5, 0.975),
) -> np.ndarray:
    """Return marginal grid quantiles without treating jittered draws as diagnostics."""
    theta = np.asarray(posterior["grid_theta"], dtype=float)
    weights = np.asarray(posterior["weights"], dtype=float)
    intervals = np.empty((theta.shape[1], len(probabilities)), dtype=float)
    for dimension in range(theta.shape[1]):
        order = np.argsort(theta[:, dimension])
        cdf = np.cumsum(weights[order])
        for probability_index, probability in enumerate(probabilities):
            intervals[dimension, probability_index] = theta[
                order[min(np.searchsorted(cdf, probability), len(order) - 1)], dimension
            ]
    return intervals


def generate_oup_mcar_partial_data_reference_posteriors(
    *,
    full_reference_path: str | os.PathLike[str] = (
        "references/reference_posteriors_v1/oup_references.h5"
    ),
    missing_fraction: float,
    reference_mask_seed: int = 987_654,
    posterior_seed: int = 23_456,
    num_reference_samples: int = 10_000,
    grid_resolution: int = 4_800,
    chunk_size: int = 200_000,
    validation_grid_resolution: int | None = 2_400,
    validation_observation_indices: tuple[int, ...] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, np.ndarray]]:
    """Generate OUP/MCAR references from exact observed-data grid posteriors.

    Full observations are loaded from the immutable original artifact.  The
    realized Point-MCAR draw mirrors estimator evaluation: one batched call
    with ``default_rng(reference_mask_seed)``.  No hidden masked state is ever
    passed to the likelihood.
    """
    if not 0.0 <= missing_fraction <= 1.0:
        raise ValueError("missing_fraction must be in [0, 1].")
    if num_reference_samples <= 0 or grid_resolution <= 1 or chunk_size <= 0:
        raise ValueError("num_reference_samples, grid_resolution, and chunk_size must be positive.")
    full_observations, _, full_metadata, _ = load_reference_posteriors_hdf5(
        full_reference_path
    )
    if full_metadata.get("problem") != "oup":
        raise ValueError("full_reference_path must point to an OUP reference artifact.")
    simulator_metadata = full_metadata["simulator"]
    simulator = OUPSimulator(
        n=int(simulator_metadata["n"]), T=float(simulator_metadata["T"]),
        var=float(simulator_metadata["var"]), y0=float(simulator_metadata["y0"]),
    )
    theta_true = np.asarray(full_observations["theta_true"], dtype=float)
    x_full = np.asarray(full_observations["x_full"], dtype=float)
    mask_generator = PointMCARMask(missing_fraction=missing_fraction)
    mask = mask_generator.generate(
        x_full, theta_true, np.random.default_rng(reference_mask_seed)
    )
    x_obs = x_full * mask
    posterior_rng = np.random.default_rng(posterior_seed)
    num_observations = x_full.shape[0]
    theta_samples = np.empty(
        (num_observations, num_reference_samples, simulator.theta_dim), dtype=float
    )
    grid_mean = np.empty((num_observations, simulator.theta_dim), dtype=float)
    grid_covariance = np.empty(
        (num_observations, simulator.theta_dim, simulator.theta_dim), dtype=float
    )
    grid_ess = np.empty(num_observations, dtype=float)
    grid_boundary_mass = np.empty(num_observations, dtype=float)
    grid_log_normalizer = np.empty(num_observations, dtype=float)
    credible_intervals = np.empty((num_observations, simulator.theta_dim, 3), dtype=float)
    runtime_seconds = np.empty(num_observations, dtype=float)
    peak_rss_mb = np.empty(num_observations, dtype=float)

    # ``ru_maxrss`` is process-wide but is still useful as a conservative peak
    # memory diagnostic on the supported POSIX execution environment.
    import resource

    final_validation: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for observation_index in range(num_observations):
        started = time.perf_counter()
        posterior = compute_oup_grid_posterior(
            x_obs[observation_index], simulator=simulator,
            grid_resolution=grid_resolution, chunk_size=chunk_size,
            mask=mask[observation_index],
        )
        theta_samples[observation_index] = sample_oup_grid_posterior(
            posterior, num_reference_samples=num_reference_samples, rng=posterior_rng
        )
        grid_mean[observation_index] = np.asarray(posterior["mean"])
        grid_covariance[observation_index] = np.asarray(posterior["cov"])
        grid_ess[observation_index] = float(posterior["ess"])
        grid_boundary_mass[observation_index] = float(posterior["boundary_mass"])
        grid_log_normalizer[observation_index] = float(posterior["log_normalizer"])
        credible_intervals[observation_index] = _oup_grid_marginal_credible_intervals(posterior)
        runtime_seconds[observation_index] = time.perf_counter() - started
        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports KiB.
        peak_rss_mb[observation_index] = (
            max_rss / (1024.0 * 1024.0)
            if os.uname().sysname == "Darwin"
            else max_rss / 1024.0
        )
        if validation_observation_indices is None or observation_index in validation_observation_indices:
            final_validation[observation_index] = (
                np.asarray(posterior["mean"]).copy(),
                np.asarray(posterior["cov"]).copy(),
                _oup_grid_marginal_credible_intervals(posterior),
            )

    if validation_grid_resolution is None:
        validation_indices = np.empty(0, dtype=np.int64)
        validation_mean_delta = np.empty((0, simulator.theta_dim), dtype=float)
        validation_cov_delta = np.empty(0, dtype=float)
        validation_ci_delta = np.empty((0, simulator.theta_dim, 3), dtype=float)
    else:
        if validation_grid_resolution <= 1:
            raise ValueError("validation_grid_resolution must exceed one.")
        if validation_observation_indices is None:
            validation_indices = np.arange(num_observations, dtype=np.int64)
        else:
            validation_indices = np.asarray(validation_observation_indices, dtype=np.int64)
        validation_mean_delta = np.empty((len(validation_indices), simulator.theta_dim), dtype=float)
        validation_cov_delta = np.empty(len(validation_indices), dtype=float)
        validation_ci_delta = np.empty((len(validation_indices), simulator.theta_dim, 3), dtype=float)
        for local_index, observation_index in enumerate(validation_indices):
            low_resolution = compute_oup_grid_posterior(
                x_obs[observation_index], simulator=simulator,
                grid_resolution=validation_grid_resolution, chunk_size=chunk_size,
                mask=mask[observation_index],
            )
            final_mean, final_cov, final_ci = final_validation[int(observation_index)]
            validation_mean_delta[local_index] = (
                np.asarray(low_resolution["mean"]) - final_mean
            )
            validation_cov_delta[local_index] = np.linalg.norm(
                np.asarray(low_resolution["cov"]) - final_cov, ord="fro"
            )
            validation_ci_delta[local_index] = (
                _oup_grid_marginal_credible_intervals(low_resolution)
                - final_ci
            )

    observations = {
        "theta_true": theta_true, "x_full": x_full, "x_obs": x_obs, "mask": mask,
    }
    metadata = {
        "problem": "oup", "missingness_mechanism": "mcar",
        "posterior_method": "deterministic_grid_exact_euler_observed_data_likelihood",
        "posterior_conditioning": "theta_given_x_obs_mask",
        "mask": mask_generator.metadata(), "reference_mask_seed": int(reference_mask_seed),
        "posterior_seed": int(posterior_seed), "full_data_reference_path": str(full_reference_path),
        "full_data_reference_version": full_metadata.get("reference_version"),
        "simulator": simulator.metadata(), "theta_scaled": False, "x_scaled": False,
        "grid_resolution": int(grid_resolution), "grid_chunk_size": int(chunk_size),
        "validation_grid_resolution": validation_grid_resolution,
        "peak_rss_unit": "MiB",
        "multi_step_transition": "finite_geometric_sum_of_euler_chain",
    }
    diagnostics = {
        "posterior_sample_mean": theta_samples.mean(axis=1),
        "posterior_sample_std": theta_samples.std(axis=1),
        "grid_mean": grid_mean, "grid_cov": grid_covariance, "grid_ess": grid_ess,
        "grid_boundary_mass": grid_boundary_mass, "grid_log_normalizer": grid_log_normalizer,
        "grid_marginal_credible_intervals": credible_intervals,
        "runtime_seconds": runtime_seconds, "peak_rss_mb": peak_rss_mb,
        "validation_observation_indices": validation_indices,
        "validation_mean_delta_low_minus_final": validation_mean_delta,
        "validation_cov_fro_delta_low_minus_final": validation_cov_delta,
        "validation_credible_interval_delta_low_minus_final": validation_ci_delta,
        "num_nonfinite_posterior_samples": np.array(
            np.count_nonzero(~np.isfinite(theta_samples)), dtype=np.int64
        ),
        "num_prior_bound_violations": np.array(
            np.count_nonzero(
                (theta_samples < simulator.prior_low) | (theta_samples > simulator.prior_high)
            ), dtype=np.int64,
        ),
    }
    return observations, metadata, {"theta_samples": theta_samples, **diagnostics}


def save_reference_posteriors_hdf5(
    path: str | os.PathLike[str],
    *,
    observations: dict[str, np.ndarray],
    theta_samples: np.ndarray,
    metadata: dict[str, Any],
    diagnostics: dict[str, np.ndarray] | None = None,
    reference_version: str = DEFAULT_REFERENCE_VERSION,
    overwrite: bool = False,
) -> None:
    """Save reference posterior samples using the GapSBI reference contract."""
    validate_reference_arrays(observations, theta_samples)
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_observations, num_reference_samples, theta_dim = theta_samples.shape
    x_dim = int(np.asarray(observations["x_full"]).shape[1])
    with h5py.File(output_path, "w") as h5:
        h5.attrs["schema_name"] = REFERENCE_SCHEMA_NAME
        h5.attrs["schema_version"] = REFERENCE_SCHEMA_VERSION
        h5.attrs["reference_version"] = reference_version
        h5.attrs["num_observations"] = int(num_observations)
        h5.attrs["num_reference_samples"] = int(num_reference_samples)
        h5.attrs["theta_dim"] = int(theta_dim)
        h5.attrs["x_dim"] = x_dim
        for key, value in metadata.items():
            h5.attrs[key] = _metadata_attr_value(value)

        obs_group = h5.create_group("observations")
        obs_group.create_dataset("theta_true", data=observations["theta_true"])
        obs_group.create_dataset("x_full", data=observations["x_full"])

        posterior_group = h5.create_group("reference_posterior")
        posterior_group.create_dataset("theta_samples", data=theta_samples)

        diagnostics_group = h5.create_group("diagnostics")
        if diagnostics is not None:
            for key, value in diagnostics.items():
                diagnostics_group.create_dataset(key, data=value)


def load_reference_posteriors_hdf5(
    path: str | os.PathLike[str],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Load and validate a reference posterior HDF5 artifact."""
    with h5py.File(path, "r") as h5:
        metadata = {key: _decode_metadata_attr(value) for key, value in h5.attrs.items()}
        observations = {
            "theta_true": h5["observations/theta_true"][...],
            "x_full": h5["observations/x_full"][...],
        }
        theta_samples = h5["reference_posterior/theta_samples"][...]
        diagnostics = {
            key: h5["diagnostics"][key][...] for key in h5["diagnostics"].keys()
        }
    validate_reference_arrays(observations, theta_samples)
    return observations, theta_samples, metadata, diagnostics


def save_partial_data_reference_posteriors_hdf5(
    path: str | os.PathLike[str],
    *,
    observations: dict[str, np.ndarray],
    theta_samples: np.ndarray,
    metadata: dict[str, Any],
    diagnostics: dict[str, np.ndarray] | None = None,
    reference_version: str = DEFAULT_PARTIAL_DATA_REFERENCE_VERSION,
    overwrite: bool = False,
) -> None:
    """Save a partial-data reference artifact using the shared v1 contract."""
    validate_partial_reference_arrays(observations, theta_samples)
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = np.asarray(theta_samples)
    with h5py.File(output_path, "w") as h5:
        h5.attrs["schema_name"] = PARTIAL_DATA_REFERENCE_SCHEMA_NAME
        h5.attrs["schema_version"] = PARTIAL_DATA_REFERENCE_SCHEMA_VERSION
        h5.attrs["reference_version"] = reference_version
        h5.attrs["num_observations"] = int(samples.shape[0])
        h5.attrs["num_reference_samples"] = int(samples.shape[1])
        h5.attrs["theta_dim"] = int(samples.shape[2])
        h5.attrs["x_dim"] = int(np.asarray(observations["x_full"]).shape[1])
        for key, value in metadata.items():
            h5.attrs[key] = _metadata_attr_value(value)

        obs_group = h5.create_group("observations")
        for key in ("theta_true", "x_full", "x_obs", "mask"):
            obs_group.create_dataset(key, data=observations[key])
        # Sparse campaigns retain their immutable original observation IDs.
        # Omitting this optional dataset preserves the existing v1 contract.
        if "reference_indices" in observations:
            reference_indices = np.asarray(observations["reference_indices"], dtype=np.int64)
            if reference_indices.shape != (samples.shape[0],):
                raise ValueError("reference_indices must have one entry per saved observation.")
            obs_group.create_dataset("reference_indices", data=reference_indices)
        h5.create_group("reference_posterior").create_dataset(
            "theta_samples", data=theta_samples
        )
        diagnostics_group = h5.create_group("diagnostics")
        if diagnostics is not None:
            for key, value in diagnostics.items():
                diagnostics_group.create_dataset(key, data=value)


def load_partial_data_reference_posteriors_hdf5(
    path: str | os.PathLike[str],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Load and validate a partial-data reference artifact using schema v1."""
    with h5py.File(path, "r") as h5:
        metadata = {key: _decode_metadata_attr(value) for key, value in h5.attrs.items()}
        if metadata.get("schema_name") != PARTIAL_DATA_REFERENCE_SCHEMA_NAME:
            raise ValueError("Not a GapSBI partial-data reference artifact.")
        observations = {
            key: h5[f"observations/{key}"][...]
            for key in ("theta_true", "x_full", "x_obs", "mask")
        }
        if "observations/reference_indices" in h5:
            observations["reference_indices"] = h5["observations/reference_indices"][...]
        theta_samples = h5["reference_posterior/theta_samples"][...]
        diagnostics = {
            key: h5["diagnostics"][key][...] for key in h5["diagnostics"].keys()
        }
    validate_partial_reference_arrays(observations, theta_samples)
    return observations, theta_samples, metadata, diagnostics


# Compatibility aliases for the original GLU/MCAR-specific public names. The
# underlying schema has always been task-agnostic, so new callers should use
# the generic functions above.
def save_glu_mcar_partial_data_reference_posteriors_hdf5(
    path: str | os.PathLike[str],
    *,
    observations: dict[str, np.ndarray],
    theta_samples: np.ndarray,
    metadata: dict[str, Any],
    diagnostics: dict[str, np.ndarray] | None = None,
    reference_version: str = DEFAULT_PARTIAL_DATA_REFERENCE_VERSION,
    overwrite: bool = False,
) -> None:
    """Backward-compatible alias for :func:`save_partial_data_reference_posteriors_hdf5`."""
    save_partial_data_reference_posteriors_hdf5(
        path, observations=observations, theta_samples=theta_samples, metadata=metadata,
        diagnostics=diagnostics, reference_version=reference_version, overwrite=overwrite,
    )


def load_glu_mcar_partial_data_reference_posteriors_hdf5(
    path: str | os.PathLike[str],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Backward-compatible alias for :func:`load_partial_data_reference_posteriors_hdf5`."""
    return load_partial_data_reference_posteriors_hdf5(path)


def plot_reference_posterior_marginals(
    reference_path: str | os.PathLike[str],
    *,
    output_dir: str | os.PathLike[str] | None = None,
    max_samples: int = 5_000,
    seed: int = 123,
    bins: int = 60,
) -> list[Path]:
    """Plot one marginal-posterior grid per reference observation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    observations, theta_samples, metadata, _ = load_reference_posteriors_hdf5(
        reference_path
    )
    reference_path = Path(reference_path)
    if output_dir is None:
        output_path = reference_path.parent / "plots"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    if theta_samples.shape[1] > max_samples:
        sample_idx = rng.choice(theta_samples.shape[1], size=max_samples, replace=False)
        plot_samples = theta_samples[:, sample_idx, :]
    else:
        plot_samples = theta_samples

    problem = str(metadata.get("problem", "reference"))
    theta_true = observations["theta_true"]
    x_full = observations["x_full"]
    show_x_markers = x_full.shape == theta_true.shape
    prior_low = _metadata_prior_bound(metadata, "prior_low", theta_samples)
    prior_high = _metadata_prior_bound(metadata, "prior_high", theta_samples)

    paths: list[Path] = []
    theta_dim = theta_samples.shape[2]
    n_cols = min(5, theta_dim)
    n_rows = int(np.ceil(theta_dim / n_cols))
    for obs_idx in range(theta_samples.shape[0]):
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.0 * n_cols, 2.4 * n_rows),
            squeeze=False,
        )
        for dim_idx, ax in enumerate(axes.ravel()):
            if dim_idx >= theta_dim:
                ax.axis("off")
                continue
            values = plot_samples[obs_idx, :, dim_idx]
            ax.hist(values, bins=bins, density=True, color="C0", alpha=0.75)
            ax.axvline(
                theta_true[obs_idx, dim_idx],
                color="C3",
                linestyle="-",
                linewidth=1.4,
                label="theta_true" if dim_idx == 0 else None,
            )
            if show_x_markers:
                ax.axvline(
                    x_full[obs_idx, dim_idx],
                    color="C2",
                    linestyle="--",
                    linewidth=1.2,
                    label="x_full" if dim_idx == 0 else None,
                )
            ax.set_xlim(prior_low[dim_idx], prior_high[dim_idx])
            ax.set_title(f"theta[{dim_idx}]", fontsize=10)
            ax.tick_params(axis="both", labelsize=8)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")
        fig.suptitle(f"{problem} reference posterior {obs_idx:02d}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        path = output_path / f"{problem}_reference_{obs_idx:02d}_marginals.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_reference_posterior_pairs(
    reference_path: str | os.PathLike[str],
    *,
    output_dir: str | os.PathLike[str] | None = None,
    max_samples: int = 10_000,
    seed: int = 123,
    bins: int = 90,
) -> list[Path]:
    """Plot posterior pair diagnostics for reference artifacts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    observations, theta_samples, metadata, _ = load_reference_posteriors_hdf5(
        reference_path
    )
    theta_dim = theta_samples.shape[2]
    if theta_dim < 2:
        return []
    reference_path = Path(reference_path)
    if output_dir is None:
        output_path = reference_path.parent / "plots"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    if theta_samples.shape[1] > max_samples:
        sample_idx = rng.choice(theta_samples.shape[1], size=max_samples, replace=False)
        plot_samples = theta_samples[:, sample_idx, :]
    else:
        plot_samples = theta_samples

    problem = str(metadata.get("problem", "reference"))
    theta_true = observations["theta_true"]
    paths: list[Path] = []
    for obs_idx in range(theta_samples.shape[0]):
        fig, axes = plt.subplots(
            theta_dim,
            theta_dim,
            figsize=(2.2 * theta_dim, 2.1 * theta_dim),
            squeeze=False,
        )
        for dim_y in range(theta_dim):
            for dim_x in range(theta_dim):
                row, col = dim_y, dim_x
                ax = axes[row, col]
                if dim_x > dim_y:
                    ax.axis("off")
                    continue
                if dim_x == dim_y:
                    ax.hist(
                        plot_samples[obs_idx, :, dim_x],
                        bins=bins,
                        density=True,
                        color="C0",
                        alpha=0.75,
                    )
                    ax.axvline(
                        theta_true[obs_idx, dim_x],
                        color="C3",
                        linewidth=1.4,
                    )
                else:
                    ax.hist2d(
                        plot_samples[obs_idx, :, dim_x],
                        plot_samples[obs_idx, :, dim_y],
                        bins=bins,
                        cmap="viridis",
                        density=True,
                    )
                    ax.scatter(
                        theta_true[obs_idx, dim_x],
                        theta_true[obs_idx, dim_y],
                        color="C3",
                        marker="x",
                        s=45,
                        linewidths=1.6,
                    )
                if dim_y == theta_dim - 1:
                    ax.set_xlabel(f"theta[{dim_x}]", fontsize=8)
                if dim_x == 0 and dim_y > 0:
                    ax.set_ylabel(f"theta[{dim_y}]", fontsize=8)
                ax.tick_params(axis="both", labelsize=7)
        fig.suptitle(f"{problem} reference posterior corner {obs_idx:02d}", fontsize=12)
        fig.tight_layout()
        path = output_path / f"{problem}_reference_{obs_idx:02d}_pair.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_reference_mcmc_traces(
    reference_path: str | os.PathLike[str],
    *,
    output_dir: str | os.PathLike[str] | None = None,
) -> list[Path]:
    """Plot stored MCMC traces when available in reference diagnostics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference_path = Path(reference_path)
    observations, theta_samples, metadata, diagnostics = load_reference_posteriors_hdf5(
        reference_path
    )
    if "mcmc_trace" not in diagnostics:
        return []
    trace = np.asarray(diagnostics["mcmc_trace"], dtype=float)
    if output_dir is None:
        output_path = reference_path.parent / "plots"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    problem = str(metadata.get("problem", "reference"))
    theta_true = observations["theta_true"]
    theta_dim = theta_samples.shape[2]
    n_cols = min(5, theta_dim)
    n_rows = int(np.ceil(theta_dim / n_cols))
    steps = np.arange(trace.shape[1])
    paths: list[Path] = []
    for obs_idx in range(trace.shape[0]):
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.2 * n_cols, 2.4 * n_rows),
            squeeze=False,
        )
        for dim_idx, ax in enumerate(axes.ravel()):
            if dim_idx >= theta_dim:
                ax.axis("off")
                continue
            for walker_idx in range(trace.shape[2]):
                ax.plot(
                    steps,
                    trace[obs_idx, :, walker_idx, dim_idx],
                    color="C0",
                    alpha=0.25,
                    linewidth=0.7,
                )
            ax.axhline(
                theta_true[obs_idx, dim_idx],
                color="C3",
                linewidth=1.1,
                label="theta_true" if dim_idx == 0 else None,
            )
            ax.set_title(f"theta[{dim_idx}]", fontsize=10)
            ax.tick_params(axis="both", labelsize=8)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")
        fig.suptitle(f"{problem} MCMC trace {obs_idx:02d}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        path = output_path / f"{problem}_reference_{obs_idx:02d}_trace.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_lotka_volterra_reference_predictives(
    reference_path: str | os.PathLike[str],
    *,
    output_dir: str | os.PathLike[str] | None = None,
    max_samples: int = 500,
    seed: int = 123,
) -> list[Path]:
    """Plot deterministic posterior predictive bands for LV reference posteriors."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    observations, theta_samples, metadata, _ = load_reference_posteriors_hdf5(
        reference_path
    )
    if metadata.get("problem") != "lotka_volterra":
        return []
    reference_path = Path(reference_path)
    if output_dir is None:
        output_path = reference_path.parent / "plots"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    simulator_metadata = metadata["simulator"]
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
    time = np.asarray(simulator_metadata["timepoints"], dtype=float)
    rng = np.random.default_rng(seed)

    paths: list[Path] = []
    for obs_idx in range(theta_samples.shape[0]):
        samples = theta_samples[obs_idx]
        if samples.shape[0] > max_samples:
            sample_idx = rng.choice(samples.shape[0], size=max_samples, replace=False)
            samples = samples[sample_idx]
        trajectories = []
        for theta in samples:
            try:
                trajectories.append(simulator._solve_states(theta).T)
            except Exception:
                continue
        if not trajectories:
            continue
        trajectories_array = np.stack(trajectories)
        q05, q50, q95 = np.quantile(trajectories_array, [0.05, 0.5, 0.95], axis=0)
        observed = observations["x_full"][obs_idx].reshape((-1, 2))

        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        labels = ("prey", "predator")
        colors = ("C0", "C2")
        for population in range(2):
            ax.fill_between(
                time,
                q05[:, population],
                q95[:, population],
                color=colors[population],
                alpha=0.18,
                label=f"{labels[population]} 90% band",
            )
            ax.plot(
                time,
                q50[:, population],
                color=colors[population],
                linewidth=1.6,
                label=f"{labels[population]} median",
            )
            ax.scatter(
                time,
                observed[:, population],
                color=colors[population],
                edgecolor="white",
                linewidth=0.4,
                s=18,
                label=f"{labels[population]} observed",
                zorder=3,
            )
        ax.set_yscale("log")
        ax.set_xlabel("time")
        ax.set_ylabel("population")
        ax.set_title(f"lotka_volterra posterior predictive {obs_idx:02d}")
        ax.legend(fontsize="x-small", ncols=2)
        fig.tight_layout()
        path = output_path / f"lotka_volterra_reference_{obs_idx:02d}_predictive.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_oup_reference_predictives(
    reference_path: str | os.PathLike[str],
    *,
    output_dir: str | os.PathLike[str] | None = None,
    max_samples: int = 1_000,
    seed: int = 123,
) -> list[Path]:
    """Plot stochastic posterior predictive bands for OUP reference posteriors."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    observations, theta_samples, metadata, _ = load_reference_posteriors_hdf5(
        reference_path
    )
    if metadata.get("problem") != "oup":
        return []
    reference_path = Path(reference_path)
    if output_dir is None:
        output_path = reference_path.parent / "plots"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    simulator_metadata = metadata["simulator"]
    simulator = OUPSimulator(
        n=int(simulator_metadata["n"]),
        T=float(simulator_metadata["T"]),
        var=float(simulator_metadata["var"]),
        y0=float(simulator_metadata["y0"]),
        prior_low=np.asarray(simulator_metadata["prior_low"], dtype=float),
        prior_high=np.asarray(simulator_metadata["prior_high"], dtype=float),
    )
    time = np.arange(simulator.n) * simulator.dt
    rng = np.random.default_rng(seed)

    paths: list[Path] = []
    for obs_idx in range(theta_samples.shape[0]):
        samples = theta_samples[obs_idx]
        if samples.shape[0] > max_samples:
            sample_idx = rng.choice(samples.shape[0], size=max_samples, replace=False)
            samples = samples[sample_idx]
        predictive_rng = np.random.default_rng(seed + 10_000 + obs_idx)
        trajectories = simulator.simulate(samples, predictive_rng)
        q05, q50, q95 = np.quantile(trajectories, [0.05, 0.5, 0.95], axis=0)
        observed = observations["x_full"][obs_idx]

        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.fill_between(
            time,
            q05,
            q95,
            color="C0",
            alpha=0.2,
            label="90% predictive band",
        )
        ax.plot(time, q50, color="C0", linewidth=1.7, label="predictive median")
        ax.plot(time, observed, color="C3", linewidth=1.5, label="observed")
        ax.scatter(time, observed, color="C3", edgecolor="white", linewidth=0.4, s=18)
        ax.set_xlabel("time")
        ax.set_ylabel("x")
        ax.set_title(f"oup posterior predictive {obs_idx:02d}")
        ax.legend(fontsize="small")
        fig.tight_layout()
        path = output_path / f"oup_reference_{obs_idx:02d}_predictive.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_glu_reference_predictives(
    reference_path: str | os.PathLike[str],
    *,
    output_dir: str | os.PathLike[str] | None = None,
    max_samples: int = 10_000,
    seed: int = 123,
) -> list[Path]:
    """Plot posterior predictive bands for GLU reference posteriors."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    observations, theta_samples, metadata, _ = load_reference_posteriors_hdf5(
        reference_path
    )
    if metadata.get("problem") != "glu":
        return []
    reference_path = Path(reference_path)
    if output_dir is None:
        output_path = reference_path.parent / "plots"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    simulator_metadata = metadata["simulator"]
    simulator_scale = float(simulator_metadata["simulator_scale"])
    rng = np.random.default_rng(seed)
    dim = theta_samples.shape[2]
    dims = np.arange(dim)

    paths: list[Path] = []
    for obs_idx in range(theta_samples.shape[0]):
        samples = theta_samples[obs_idx]
        if samples.shape[0] > max_samples:
            sample_idx = rng.choice(samples.shape[0], size=max_samples, replace=False)
            samples = samples[sample_idx]
        predictive_rng = np.random.default_rng(seed + 20_000 + obs_idx)
        x_rep = samples + predictive_rng.normal(
            loc=0.0,
            scale=simulator_scale,
            size=samples.shape,
        )
        q05, q50, q95 = np.quantile(x_rep, [0.05, 0.5, 0.95], axis=0)
        observed = observations["x_full"][obs_idx]

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.fill_between(
            dims,
            q05,
            q95,
            color="C0",
            alpha=0.22,
            step="mid",
            label="90% predictive band",
        )
        ax.plot(dims, q50, color="C0", marker="o", linewidth=1.5, label="predictive median")
        ax.scatter(
            dims,
            observed,
            color="C3",
            edgecolor="white",
            linewidth=0.5,
            s=34,
            label="observed",
            zorder=3,
        )
        ax.set_xlabel("dimension")
        ax.set_ylabel("x")
        ax.set_xticks(dims)
        ax.set_title(f"glu posterior predictive {obs_idx:02d}")
        ax.legend(fontsize="small")
        fig.tight_layout()
        path = output_path / f"glu_reference_{obs_idx:02d}_predictive.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_glm_reference_predictives(
    reference_path: str | os.PathLike[str],
    *,
    output_dir: str | os.PathLike[str] | None = None,
    max_samples: int = 10_000,
    seed: int = 123,
) -> list[Path]:
    """Plot posterior predictive spike-probability bands for GLM references."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    observations, theta_samples, metadata, _ = load_reference_posteriors_hdf5(
        reference_path
    )
    if metadata.get("problem") != "glm":
        return []
    reference_path = Path(reference_path)
    if output_dir is None:
        output_path = reference_path.parent / "plots"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    simulator_metadata = metadata["simulator"]
    simulator = GLMSimulator(
        dim=int(simulator_metadata["dim"]),
        duration=int(simulator_metadata["duration"]),
        stimulus_seed=int(simulator_metadata["stimulus_seed"]),
        summary="raw",
    )
    rng = np.random.default_rng(seed)
    time = np.arange(simulator.duration)

    paths: list[Path] = []
    for obs_idx in range(theta_samples.shape[0]):
        samples = theta_samples[obs_idx]
        if samples.shape[0] > max_samples:
            sample_idx = rng.choice(samples.shape[0], size=max_samples, replace=False)
            samples = samples[sample_idx]
        probabilities = simulator._sigmoid(samples @ simulator.design_matrix.T)
        q05, q50, q95 = np.quantile(probabilities, [0.05, 0.5, 0.95], axis=0)
        observed = observations["x_full"][obs_idx].astype(float)
        expected_count = probabilities.sum(axis=1)

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(8.0, 5.6),
            gridspec_kw={"height_ratios": [2.0, 1.0]},
        )
        ax = axes[0]
        ax.fill_between(
            time,
            q05,
            q95,
            color="C0",
            alpha=0.22,
            label="90% probability band",
        )
        ax.plot(time, q50, color="C0", linewidth=1.5, label="probability median")
        spike_times = time[observed == 1.0]
        if spike_times.size > 0:
            ax.scatter(
                spike_times,
                np.full(spike_times.shape, 1.03),
                color="C3",
                marker="|",
                s=80,
                linewidth=1.4,
                label="observed spike",
                clip_on=False,
            )
        ax.set_ylim(-0.03, 1.08)
        ax.set_xlabel("time bin")
        ax.set_ylabel("spike probability")
        ax.set_title(f"glm posterior predictive probabilities {obs_idx:02d}")
        ax.legend(fontsize="small")

        ax_count = axes[1]
        ax_count.hist(expected_count, bins=40, color="C0", alpha=0.75)
        ax_count.axvline(
            observed.sum(),
            color="C3",
            linewidth=1.5,
            label="observed spike count",
        )
        ax_count.set_xlabel("posterior expected spike count")
        ax_count.set_ylabel("count")
        ax_count.legend(fontsize="small")
        fig.tight_layout()
        path = output_path / f"glm_reference_{obs_idx:02d}_predictive.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def validate_reference_arrays(
    observations: dict[str, np.ndarray],
    theta_samples: np.ndarray,
) -> None:
    """Validate the in-memory reference posterior contract."""
    if "theta_true" not in observations or "x_full" not in observations:
        raise ValueError("observations must contain theta_true and x_full.")
    theta_true = np.asarray(observations["theta_true"])
    x_full = np.asarray(observations["x_full"])
    samples = np.asarray(theta_samples)
    if theta_true.ndim != 2:
        raise ValueError("observations/theta_true must be 2D.")
    if x_full.ndim != 2:
        raise ValueError("observations/x_full must be 2D.")
    if samples.ndim != 3:
        raise ValueError("reference_posterior/theta_samples must be 3D.")
    if theta_true.shape[0] != x_full.shape[0] or theta_true.shape[0] != samples.shape[0]:
        raise ValueError("reference arrays must have matching observation counts.")
    if theta_true.shape[1] != samples.shape[2]:
        raise ValueError("theta_true dimension must match theta_samples dimension.")
    for name, array in {
        "theta_true": theta_true,
        "x_full": x_full,
        "theta_samples": samples,
    }.items():
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")


def validate_partial_reference_arrays(
    observations: dict[str, np.ndarray],
    theta_samples: np.ndarray,
) -> None:
    """Validate the task-agnostic partial-data reference contract."""
    required = {"theta_true", "x_full", "x_obs", "mask"}
    if not required.issubset(observations):
        raise ValueError(f"observations must contain {sorted(required)}.")
    validate_reference_arrays(observations, theta_samples)
    x_full = np.asarray(observations["x_full"])
    x_obs = np.asarray(observations["x_obs"])
    mask = np.asarray(observations["mask"])
    if x_obs.shape != x_full.shape or mask.shape != x_full.shape:
        raise ValueError("x_obs and mask must match x_full shape.")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("mask must be binary with 1=observed and 0=missing.")
    if not np.array_equal(x_obs, x_full * mask):
        raise ValueError("x_obs must equal x_full * mask.")


def validate_glu_mcar_partial_data_reference_arrays(
    observations: dict[str, np.ndarray],
    theta_samples: np.ndarray,
) -> None:
    """Backward-compatible alias for :func:`validate_partial_reference_arrays`."""
    validate_partial_reference_arrays(observations, theta_samples)


def _metadata_attr_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str | int | float | bool):
        return value
    return json.dumps(value)


def _decode_metadata_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _metadata_prior_bound(
    metadata: dict[str, Any],
    key: str,
    theta_samples: np.ndarray,
) -> np.ndarray:
    simulator = metadata.get("simulator", {})
    if isinstance(simulator, dict) and key in simulator:
        return np.asarray(simulator[key], dtype=float)
    values = theta_samples.reshape(-1, theta_samples.shape[-1])
    if key == "prior_low":
        return values.min(axis=0)
    if key == "prior_high":
        return values.max(axis=0)
    raise ValueError(f"Unknown prior bound key: {key}")
