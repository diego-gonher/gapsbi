from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import truncnorm

from gapsbi.simulators import GLMSimulator, GLUSimulator, OUPSimulator


REFERENCE_SCHEMA_NAME = "gapsbi_reference_posteriors"
REFERENCE_SCHEMA_VERSION = 1
DEFAULT_REFERENCE_VERSION = "reference_posteriors_v1"


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
        diagnostics.update(
            compute_oup_grid_resolution_diagnostics(
                x_full,
                simulator=simulator,
                reference_posteriors=final_posteriors,
                validation_grid_resolutions=validation_grid_resolutions,
            )
        )
    if spotcheck_grid_resolution is not None:
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
    prior_bound: float = 2.0,
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

    for local_idx, obs_idx in enumerate(validation_observation_indices):
        if obs_idx < 0 or obs_idx >= num_observations:
            continue
        reference_mean = posterior_sample_mean[obs_idx]
        reference_cov = np.cov(theta_samples[obs_idx], rowvar=False)
        for ensemble_idx in range(validation_num_ensembles):
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
        "prior_low": simulator.prior_low,
        "prior_high": simulator.prior_high,
        "num_prior_bound_violations": np.array(
            np.count_nonzero(
                (theta_samples < simulator.prior_low)
                | (theta_samples > simulator.prior_high)
            ),
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
    log_prob = np.full(theta_array.shape[0], -np.inf, dtype=float)
    in_prior = np.all(
        (theta_array >= simulator.prior_low)
        & (theta_array <= simulator.prior_high),
        axis=1,
    )
    if np.any(in_prior):
        log_prob[in_prior] = glm_log_likelihood(
            theta_array[in_prior],
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
    sampler.run_mcmc(initial_state, burn_in_steps, progress=False)
    sampler.reset()
    sampler.run_mcmc(None, production_steps, progress=False)
    chain = sampler.get_chain()
    acceptance_fraction = float(np.mean(sampler.acceptance_fraction))
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
    """Find a bounded GLM posterior mode for walker initialization."""
    starts = [
        np.zeros(simulator.theta_dim, dtype=float),
        np.full(simulator.theta_dim, 0.5, dtype=float),
        np.full(simulator.theta_dim, -0.5, dtype=float),
    ]
    bounds = list(zip(simulator.prior_low, simulator.prior_high, strict=True))

    def objective(theta: np.ndarray) -> float:
        return -float(glm_log_prob(theta, x_full, simulator))

    best_x = starts[0]
    best_fun = objective(best_x)
    for start in starts:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
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
    """Initialize GLM walkers near a mode while respecting prior support."""
    walkers = rng.normal(
        loc=center,
        scale=scale,
        size=(num_walkers, simulator.theta_dim),
    )
    invalid = np.any(
        (walkers <= simulator.prior_low)
        | (walkers >= simulator.prior_high),
        axis=1,
    )
    while np.any(invalid):
        walkers[invalid] = rng.normal(
            loc=center,
            scale=scale,
            size=(int(np.sum(invalid)), simulator.theta_dim),
        )
        walkers = np.clip(
            walkers,
            simulator.prior_low + 1e-6,
            simulator.prior_high - 1e-6,
        )
        invalid = np.any(
            (walkers <= simulator.prior_low)
            | (walkers >= simulator.prior_high),
            axis=1,
        )
    return walkers


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


def compute_oup_grid_posterior(
    x_full: np.ndarray,
    *,
    simulator: OUPSimulator | None = None,
    grid_resolution: int,
    chunk_size: int = 200_000,
) -> dict[str, np.ndarray | float | int]:
    """Approximate an OUP posterior over a cell-centered prior grid."""
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
        log_weights[start:stop] = oup_log_likelihood(
            grid_theta[start:stop],
            x_full,
            simulator,
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
    """Save reference posterior samples using the GAPSBI reference contract."""
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
    """Plot 2D posterior histograms for two-parameter reference artifacts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    observations, theta_samples, metadata, _ = load_reference_posteriors_hdf5(
        reference_path
    )
    if theta_samples.shape[2] != 2:
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
    prior_low = _metadata_prior_bound(metadata, "prior_low", theta_samples)
    prior_high = _metadata_prior_bound(metadata, "prior_high", theta_samples)
    paths: list[Path] = []
    for obs_idx in range(theta_samples.shape[0]):
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        hist = ax.hist2d(
            plot_samples[obs_idx, :, 0],
            plot_samples[obs_idx, :, 1],
            bins=bins,
            range=[
                [prior_low[0], prior_high[0]],
                [prior_low[1], prior_high[1]],
            ],
            cmap="viridis",
            density=True,
        )
        fig.colorbar(hist[3], ax=ax, label="density")
        ax.scatter(
            theta_true[obs_idx, 0],
            theta_true[obs_idx, 1],
            color="C3",
            marker="x",
            s=70,
            linewidths=2.0,
            label="theta_true",
        )
        ax.set_xlim(prior_low[0], prior_high[0])
        ax.set_ylim(prior_low[1], prior_high[1])
        ax.set_xlabel("theta[0]")
        ax.set_ylabel("theta[1]")
        ax.set_title(f"{problem} reference posterior {obs_idx:02d}")
        ax.legend(loc="best")
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
