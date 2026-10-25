"""Resumable blocked joint-MCMC pilot for LV/MNAR hard-support cases."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from gapsbi.masks import LotkaVolterraLogTotalMNARMask
from gapsbi.references import (
    load_reference_posteriors_hdf5,
    lotka_volterra_log_prob,
    lotka_volterra_log_total_mnar_log_probability,
    lotka_volterra_mnar_support_tilt,
)
from gapsbi.simulators import LotkaVolterraSimulator

ROOT = Path("outputs/diagnostics/lv_mnar_joint_mcmc_pilot")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=_json_default), encoding="utf-8")
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def make_simulator(metadata: dict) -> LotkaVolterraSimulator:
    values = metadata["simulator"]
    return LotkaVolterraSimulator(
        num_timepoints=values["num_timepoints"], days=values["days"],
        observation_noise_scale=values["observation_noise_scale"],
        initial_state=tuple(values["initial_state"]),
        prior_log_mean=np.asarray(values["prior_log_mean"]),
        prior_log_std=np.asarray(values["prior_log_std"]), max_state=values["max_state"],
        ode_rtol=values["ode_rtol"], ode_atol=values["ode_atol"],
    )


def mean_log_observations(z: np.ndarray, simulator: LotkaVolterraSimulator) -> np.ndarray:
    states = simulator._solve_states(np.exp(z)).T.reshape(-1)
    return np.log(np.clip(states, 1e-10, simulator.max_state))


def completion_from_mu(x_obs: np.ndarray, mask: np.ndarray, mu: np.ndarray, xi: np.ndarray, sigma: float) -> np.ndarray:
    """Construct complete noisy data; xi contains missing coordinates only."""
    missing = np.asarray(mask, dtype=bool) == 0
    if xi.shape != (int(missing.sum()),):
        raise ValueError("xi must have one entry per missing coordinate.")
    complete = np.asarray(x_obs, dtype=float).copy()
    complete[missing] = np.exp(mu[missing] + sigma * xi)
    return complete


def latent_log_likelihood(mu: np.ndarray, xi: np.ndarray, x_obs: np.ndarray, mask: np.ndarray, *, sigma: float, missing_fraction: float) -> float:
    """ESS's slice likelihood, log p(mask | x_complete), without N(0,I)."""
    complete = completion_from_mu(x_obs, mask, mu, xi, sigma)
    return lotka_volterra_log_total_mnar_log_probability(complete, mask, missing_fraction=missing_fraction)


def joint_log_probability(z: np.ndarray, xi: np.ndarray, x_obs: np.ndarray, mask: np.ndarray, simulator: LotkaVolterraSimulator, *, missing_fraction: float, mu: np.ndarray | None = None) -> tuple[float, np.ndarray | None]:
    """Exact log p(z,xi | x_obs,m), modulo constants independent of z,xi.

    y_mis = mu(z) + sigma xi has a sigma-only Jacobian, so it is absorbed in
    the constant.  N(0,I) occurs here once and not in the ESS slice term.
    """
    try:
        current_mu = mean_log_observations(z, simulator) if mu is None else mu
        observed = float(lotka_volterra_log_prob(z, x_obs, simulator, mask=mask))
        mask_term = latent_log_likelihood(
            current_mu, xi, x_obs, mask, sigma=simulator.observation_noise_scale,
            missing_fraction=missing_fraction,
        )
        if not np.isfinite(observed) or not np.isfinite(mask_term):
            return -np.inf, current_mu
        return float(observed - .5 * np.dot(xi, xi) + mask_term), current_mu
    except Exception:
        return -np.inf, None


def elliptical_slice_latent(xi: np.ndarray, mu: np.ndarray, x_obs: np.ndarray, mask: np.ndarray, *, simulator: LotkaVolterraSimulator, missing_fraction: float, rng: np.random.Generator, max_bracket_steps: int) -> tuple[np.ndarray, float, int]:
    """Exact ESS update with a bounded, explicit bracket-exhaustion failure."""
    current = latent_log_likelihood(mu, xi, x_obs, mask, sigma=simulator.observation_noise_scale, missing_fraction=missing_fraction)
    if not np.isfinite(current):
        raise RuntimeError("ESS was asked to update an infeasible latent state.")
    log_slice = current + np.log(rng.random())
    direction = rng.normal(size=xi.size)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    lower, upper = angle - 2.0 * np.pi, angle
    for evaluations in range(1, max_bracket_steps + 1):
        proposal = xi * np.cos(angle) + direction * np.sin(angle)
        proposal_log_like = latent_log_likelihood(mu, proposal, x_obs, mask, sigma=simulator.observation_noise_scale, missing_fraction=missing_fraction)
        if np.isfinite(proposal_log_like) and proposal_log_like >= log_slice:
            return proposal, proposal_log_like, evaluations
        if angle < 0.0:
            lower = angle
        else:
            upper = angle
        angle = rng.uniform(lower, upper)
    raise RuntimeError(f"ESS bracket safeguard exhausted after {max_bracket_steps} proposals")


def initialize_feasible_latent(z: np.ndarray, x_obs: np.ndarray, mask: np.ndarray, simulator: LotkaVolterraSimulator, *, missing_fraction: float, rng: np.random.Generator, max_attempts: int = 20_000) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """Use the tilt solely to find an initial state; MCMC targets no tilt."""
    mu = mean_log_observations(z, simulator)
    tilt = lotka_volterra_mnar_support_tilt(np.exp(mu), x_obs, mask, missing_fraction=missing_fraction, sigma=simulator.observation_noise_scale, safety_margin=.30)
    dimension = int(np.sum(np.asarray(mask, dtype=bool) == 0))
    for attempt in range(1, max_attempts + 1):
        xi = rng.normal(tilt / simulator.observation_noise_scale, 1.0, size=dimension)
        target, _ = joint_log_probability(z, xi, x_obs, mask, simulator, missing_fraction=missing_fraction, mu=mu)
        if np.isfinite(target):
            return xi, mu, target, tilt, attempt
    raise RuntimeError("Could not find a feasible latent initialization without hidden data.")


def autocorrelation_time(chain: np.ndarray) -> np.ndarray:
    try:
        from emcee.autocorr import integrated_time
        return np.asarray(integrated_time(chain[:, None, :], tol=0), dtype=float)
    except Exception:
        return np.full(chain.shape[-1], np.nan)


def coupled_latent_parameter_proposal(xi: np.ndarray, old_mu: np.ndarray, new_mu: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Keep the completed missing log observations fixed under a z proposal.

    xi' = xi + (mu(z)-mu(z'))/sigma on missing coordinates.  The map is
    triangular in (z, xi), hence has Jacobian one, and is exactly reversible.
    """
    missing = np.asarray(mask, dtype=bool) == 0
    return xi + (old_mu[missing] - new_mu[missing]) / sigma


def run_chain(*, chain_index: int, initial_z: np.ndarray, x_obs: np.ndarray, mask: np.ndarray, simulator: LotkaVolterraSimulator, missing_fraction: float, seed: int, burn_in: int, production: int, mh_scale: float, proposal_cholesky: np.ndarray, max_ess_bracket_steps: int, directory: Path, extend_production: int = 0, parameter_update: str = "full", adapt_covariance_warmup: bool = False) -> None:
    output = directory / f"chain_{chain_index:02d}.npz"
    summary = directory / f"chain_{chain_index:02d}.json"
    completed = output.exists() and summary.exists() and json.loads(summary.read_text()).get("status") == "completed"
    if completed and (not extend_production or json.loads(summary.read_text()).get("configuration", {}).get("extend_production") == extend_production):
        return
    rng = np.random.default_rng(seed)
    previous_z = previous_xi = previous_log_joint = None
    if completed:
        previous = np.load(output)
        previous_z, previous_xi, previous_log_joint = previous["log_theta"], previous["latent_xi"], previous["log_joint"]
        z, xi = previous_z[-1].copy(), previous_xi[-1].copy()
        current, mu = joint_log_probability(z, xi, x_obs, mask, simulator, missing_fraction=missing_fraction)
        if not np.isfinite(current):
            raise RuntimeError("Persisted chain ends at an infeasible joint state.")
        tilt, init_attempts, burn_in, production = np.nan, 0, 0, extend_production
        initial = {"resumed_from": str(output), "log_theta": z, "joint_log_probability": current, "seed": seed}
    else:
        xi, mu, current, tilt, init_attempts = initialize_feasible_latent(initial_z, x_obs, mask, simulator, missing_fraction=missing_fraction, rng=rng)
        z = np.asarray(initial_z, dtype=float).copy()
        initial = {"log_theta": z, "joint_log_probability": current, "tilt": tilt, "initialization_attempts": init_attempts, "seed": seed}
    total = burn_in + production
    samples_z, samples_xi, samples_logp = [], [], []
    warmup_z: list[np.ndarray] = []
    active_cholesky = proposal_cholesky.copy()
    accepted = hard_rejections = ode_failures = ess_evaluations = 0
    atomic_json(summary, {"status": "running", "chain_index": chain_index, "configuration": {"burn_in": burn_in, "production": production, "mh_scale": mh_scale, "max_ess_bracket_steps": max_ess_bracket_steps, "parameter_update": parameter_update, "adapt_covariance_warmup": adapt_covariance_warmup}, "initialization": initial})
    started = time.perf_counter()
    for iteration in range(total):
        xi, _, steps = elliptical_slice_latent(xi, mu, x_obs, mask, simulator=simulator, missing_fraction=missing_fraction, rng=rng, max_bracket_steps=max_ess_bracket_steps)
        ess_evaluations += steps
        current, _ = joint_log_probability(z, xi, x_obs, mask, simulator, missing_fraction=missing_fraction, mu=mu)
        proposal_z = z + mh_scale * (active_cholesky @ rng.normal(size=4))
        try:
            proposal_mu = mean_log_observations(proposal_z, simulator)
        except Exception:
            proposal_mu = None
        if proposal_mu is None:
            ode_failures += 1
            proposed = -np.inf
            proposal_xi = xi
        else:
            proposal_xi = (coupled_latent_parameter_proposal(xi, mu, proposal_mu, mask, simulator.observation_noise_scale)
                           if parameter_update == "coupled" else xi)
            proposed, _ = joint_log_probability(proposal_z, proposal_xi, x_obs, mask, simulator, missing_fraction=missing_fraction, mu=proposal_mu)
        if not np.isfinite(proposed):
            hard_rejections += 1
        elif np.log(rng.random()) < proposed - current:
            z, xi, mu, current = proposal_z, proposal_xi, proposal_mu, proposed
            accepted += 1
        if iteration < burn_in and adapt_covariance_warmup:
            warmup_z.append(z.copy())
            if iteration >= 199 and (iteration + 1) % 100 == 0:
                covariance = np.cov(np.asarray(warmup_z), rowvar=False)
                try:
                    active_cholesky = np.linalg.cholesky(covariance + 1e-8 * np.eye(4))
                except np.linalg.LinAlgError:
                    pass
        if iteration >= burn_in:
            samples_z.append(z.copy()); samples_xi.append(xi.copy()); samples_logp.append(current)
    log_theta, latent, log_joint = np.asarray(samples_z), np.asarray(samples_xi), np.asarray(samples_logp)
    if previous_z is not None:
        log_theta = np.concatenate([previous_z, log_theta])
        latent = np.concatenate([previous_xi, latent])
        log_joint = np.concatenate([previous_log_joint, log_joint])
    tau = autocorrelation_time(log_theta)
    atomic_npz(output, log_theta=log_theta, latent_xi=latent, log_joint=log_joint)
    atomic_json(summary, {"status": "completed", "chain_index": chain_index, "configuration": {"burn_in": burn_in, "production": production, "mh_scale": mh_scale, "max_ess_bracket_steps": max_ess_bracket_steps, "extend_production": extend_production, "parameter_update": parameter_update, "adapt_covariance_warmup": adapt_covariance_warmup}, "initialization": initial, "mh_acceptance_rate": accepted / total, "hard_support_rejections": hard_rejections, "ordinary_posterior_rejections": total - hard_rejections - accepted, "ode_failures": ode_failures, "ode_solves_for_parameter_proposals": total, "ess_likelihood_evaluations": ess_evaluations, "mean_ess_evaluations_per_latent_update": ess_evaluations / total, "final_proposal_covariance": active_cholesky @ active_cholesky.T, "autocorrelation_time_heuristic": tau, "heuristic_ess": len(log_theta) / tau, "posterior_mean_raw": np.exp(log_theta).mean(axis=0), "posterior_covariance_raw": np.cov(np.exp(log_theta), rowvar=False), "posterior_quantiles_raw": np.quantile(np.exp(log_theta), [.05, .5, .95], axis=0), "runtime_seconds": time.perf_counter() - started})


def analyze(directory: Path) -> dict:
    # Import only after sampling: ArviZ creates a user cache on import, while
    # the core sampler and its unit tests do not require that side effect.
    import arviz as az
    chains = [np.load(path)["log_theta"] for path in sorted(directory.glob("chain_*.npz"))]
    if len(chains) < 2:
        return {"status": "incomplete", "completed_chains": len(chains)}
    # Extensions can leave independently persisted chains at different lengths.
    # R-hat requires equal draw counts, so use the common late segment only.
    common_draws = min(chain.shape[0] for chain in chains)
    stacked = np.asarray([chain[-common_draws:] for chain in chains])
    raw = np.exp(stacked)
    summaries = [json.loads(path.read_text()) for path in sorted(directory.glob("chain_*.json"))]
    # ArviZ's ndarray API is univariate; each column below is formed from the
    # four genuinely independent joint chains, never from interacting walkers.
    rhat = np.array([az.rhat(stacked[:, :, parameter], method="rank") for parameter in range(4)])
    bulk = np.array([az.ess(stacked[:, :, parameter], method="bulk") for parameter in range(4)])
    tail = np.array([az.ess(stacked[:, :, parameter], method="tail") for parameter in range(4)])
    diagnostics = {"status": "completed", "completed_chains": len(chains), "common_late_draws_for_cross_chain_diagnostics": common_draws, "rank_rhat_log_theta": rhat, "bulk_ess_log_theta": bulk, "tail_ess_log_theta": tail, "posterior_mean_raw_by_chain": raw.mean(axis=1), "posterior_quantiles_raw_by_chain": np.quantile(raw, [.05, .5, .95], axis=1), "max_mean_difference_raw": float(np.max(np.ptp(raw.mean(axis=1), axis=0))), "chain_summaries": summaries}
    atomic_json(directory / "analysis.json", diagnostics)
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-index", type=int, default=2)
    parser.add_argument("--missing-fraction", type=float, default=.5)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--burn-in", type=int, default=300)
    parser.add_argument("--production", type=int, default=1000)
    parser.add_argument("--mh-scale", type=float, default=.65)
    parser.add_argument("--max-ess-bracket-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=94601)
    parser.add_argument("--extend-production", type=int, default=0)
    parser.add_argument("--chain-index", type=int, default=None, help="Run one chain only; useful for supervised resumption.")
    parser.add_argument("--parameter-update", choices=("full", "coupled"), default="full")
    parser.add_argument("--adapt-covariance-warmup", action="store_true")
    parser.add_argument("--output-tag", type=str, default="")
    args = parser.parse_args()
    if args.missing_fraction != .5:
        raise ValueError("This focused pilot is deliberately limited to 50% MNAR.")
    observations, _, metadata, _ = load_reference_posteriors_hdf5("references/reference_posteriors_v1/lotka_volterra_references.h5")
    simulator = make_simulator(metadata)
    masks = LotkaVolterraLogTotalMNARMask(args.missing_fraction).generate(observations["x_full"], observations["theta_true"], np.random.default_rng(987654))
    mask = masks[args.reference_index]
    x_obs = observations["x_full"][args.reference_index] * mask
    source = Path("outputs/reference_generation/lv_mnar/eps050") / f"reference_{args.reference_index:02d}"
    source_chains = [np.load(source / f"ensemble_{ensemble}.npz")["log_theta_chain"] for ensemble in range(2)]
    flattened = np.concatenate([value[-1000:].reshape(-1, 4) for value in source_chains])
    proposal_cholesky = np.linalg.cholesky(np.cov(flattened, rowvar=False) + 1e-10 * np.eye(4))
    starts = flattened[np.linspace(0, len(flattened) - 1, args.chains, dtype=int)]
    directory = ROOT / f"eps050_ref{args.reference_index:02d}" / args.output_tag if args.output_tag else ROOT / f"eps050_ref{args.reference_index:02d}"
    atomic_json(directory / "manifest.json", {"status": "running", "problem": "lotka_volterra", "mechanism": "mnar", "reference_index": args.reference_index, "proposal_source": str(source), "target": "p(z,xi|x_obs,m)", "mask_seed": 987654, "configuration": vars(args)})
    for chain_index, initial_z in enumerate(starts):
        if args.chain_index is not None and chain_index != args.chain_index:
            continue
        run_chain(chain_index=chain_index, initial_z=initial_z, x_obs=x_obs, mask=mask, simulator=simulator, missing_fraction=args.missing_fraction, seed=int(np.random.SeedSequence([args.seed, chain_index, args.extend_production]).generate_state(1)[0]), burn_in=args.burn_in, production=args.production, mh_scale=args.mh_scale, proposal_cholesky=proposal_cholesky, max_ess_bracket_steps=args.max_ess_bracket_steps, directory=directory, extend_production=args.extend_production, parameter_update=args.parameter_update, adapt_covariance_warmup=args.adapt_covariance_warmup)
    analysis = analyze(directory) if args.chain_index is None else {"status": "running", "completed_chains": len(list(directory.glob("chain_*.npz")))}
    atomic_json(directory / "manifest.json", {"status": analysis["status"], "problem": "lotka_volterra", "mechanism": "mnar", "reference_index": args.reference_index, "proposal_source": str(source), "target": "p(z,xi|x_obs,m)", "mask_seed": 987654, "configuration": vars(args), "analysis": analysis})


if __name__ == "__main__":
    main()
