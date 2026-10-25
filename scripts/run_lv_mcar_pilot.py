"""Run resumable, independently replicated LV/MCAR sampler pilots.

This script deliberately writes diagnostics after every case.  It does not
create reference artifacts and does not store posterior draws: only summaries,
convergence diagnostics, provenance, and failure records are persisted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from gapsbi.masks import LotkaVolterraLogTotalMNARMask, LotkaVolterraTimeBlockMCARMask
from gapsbi.references import (
    load_reference_posteriors_hdf5,
    lotka_volterra_log_prob,
    run_lotka_volterra_emcee_reference,
)
from gapsbi.simulators import LotkaVolterraSimulator


def parse_case(value: str) -> tuple[float, int]:
    try:
        fraction_text, index_text = value.split(":", maxsplit=1)
        fraction, index = float(fraction_text), int(index_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("case must have form FRACTION:INDEX, e.g. 0.50:1") from error
    if fraction not in (0.10, 0.25, 0.50) or index < 0 or index >= 10:
        raise argparse.ArgumentTypeError("fraction must be 0.10, 0.25, or 0.50; index must be in [0, 9]")
    return fraction, index


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True, type=parse_case,
                        help="May be repeated; e.g. --case 0.50:1")
    parser.add_argument("--mechanism", choices=("mcar", "mnar"), default="mcar")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostics/lv_mcar_pilot"))
    parser.add_argument("--full-reference", type=Path,
                        default=Path("references/reference_posteriors_v1/lotka_volterra_references.h5"))
    parser.add_argument("--reference-mask-seed", type=int, default=987_654)
    parser.add_argument("--posterior-seed", type=int, default=810_000)
    parser.add_argument("--num-ensembles", type=int, default=2)
    parser.add_argument("--num-walkers", type=int, default=32)
    parser.add_argument("--burn-in-steps", type=int, default=500)
    parser.add_argument("--production-steps", type=int, default=1_500)
    parser.add_argument("--initial-scale", type=float, default=0.30)
    parser.add_argument("--diagnostic-walker-index", type=int, default=0,
                        help="Fixed walker retained per independent ensemble for ArviZ diagnostics.")
    parser.add_argument("--extended-production-steps", type=int, default=0,
                        help="Optional one additional, longer independent ensemble per case.")
    parser.add_argument("--extended-burn-in-steps", type=int, default=None,
                        help="Burn-in for the optional extended ensemble (defaults to --burn-in-steps).")
    parser.add_argument("--append-extended-run", action="store_true",
                        help="Append one extended run to an already completed case; primary ensembles stay untouched.")
    parser.add_argument("--initialization", choices=("map", "high-tail", "constrained-high-tail", "overdispersed"), default="map",
                        help="Walker initialization regime for a one-ensemble targeted follow-up.")
    parser.add_argument("--previous-pilot", type=Path, default=None,
                        help="Prior pilot JSON; required for --initialization high-tail.")
    parser.add_argument("--store-chains", action="store_true",
                        help="Persist log-parameter and log-posterior production chains in a compressed NPZ file.")
    parser.add_argument("--screen-initialization", action="store_true",
                        help="Use the MAP/Hessian local-density screen before MCMC starts.")
    parser.add_argument("--rerun-incomplete", action="store_true",
                        help="Retry a saved failed/running case; completed cases are always skipped.")
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=_json_default), encoding="utf-8")
    temporary.replace(path)


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def _simulator(metadata: dict[str, Any]) -> LotkaVolterraSimulator:
    spec = metadata["simulator"]
    return LotkaVolterraSimulator(
        num_timepoints=int(spec["num_timepoints"]), days=float(spec["days"]),
        observation_noise_scale=float(spec["observation_noise_scale"]),
        initial_state=tuple(spec["initial_state"]), prior_log_mean=np.asarray(spec["prior_log_mean"]),
        prior_log_std=np.asarray(spec["prior_log_std"]), max_state=float(spec["max_state"]),
        ode_rtol=float(spec["ode_rtol"]), ode_atol=float(spec["ode_atol"]),
    )


def _seed(base: int, fraction: float, index: int, ensemble: int, extended: bool = False) -> int:
    return int(np.random.SeedSequence([base, round(fraction * 100), index, ensemble, int(extended)]).generate_state(1)[0])


def _targeted_initialization(
    args: argparse.Namespace, simulator: LotkaVolterraSimulator, x_obs: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if args.initialization == "map":
        return None, {"regime": "map", "description": "Partial-data MAP returned by the sampler's multistart optimizer."}
    if args.initialization == "overdispersed":
        return simulator.prior_log_mean.copy(), {
            "regime": "overdispersed", "center_log_theta": simulator.prior_log_mean.copy(),
            "description": "Prior-mean center with the requested broad walker scale.",
        }
    if args.initialization == "constrained-high-tail":
        # The preliminary run stored only a representative walker, which did
        # not retain the rare high-alpha excursion. Locate a defensible start
        # in that region by optimizing the *partial-data* target with alpha
        # constrained to a clearly separated high-tail value. This does not
        # use the hidden simulation parameter.
        from scipy.optimize import minimize

        alpha_floor = float(np.log(1.3))
        start = simulator.prior_log_mean.copy()
        start[0] = np.log(1.5)

        def objective(value: np.ndarray) -> float:
            score = float(lotka_volterra_log_prob(value, x_obs, simulator, mask=mask))
            return -score if np.isfinite(score) else 1e100

        optimization = minimize(
            objective, start, method="Powell",
            bounds=[(alpha_floor, np.log(4.0)), (-8.0, 0.0), (-4.0, 2.0), (-8.0, 0.0)],
            options={"maxiter": 800, "xtol": 1e-5, "ftol": 1e-5},
        )
        center = np.asarray(optimization.x, dtype=float)
        center_log_prob = -float(optimization.fun)
        if not optimization.success or not np.isfinite(center_log_prob):
            raise RuntimeError("Constrained high-tail optimization failed to find a finite start.")
        return center, {
            "regime": "constrained_high_tail", "center_log_theta": center,
            "center_log_prob": center_log_prob, "alpha_lower_bound": float(np.exp(alpha_floor)),
            "optimization_method": "Powell on the partial-data log posterior",
            "optimization_message": str(optimization.message),
        }
    if args.previous_pilot is None:
        raise ValueError("--previous-pilot is required for --initialization high-tail.")
    previous = json.loads(args.previous_pilot.read_text(encoding="utf-8"))
    trajectories = previous.get("independent_representative_log_theta_chains", [])
    if not trajectories:
        raise ValueError("Previous pilot contains no persisted representative log-parameter trajectories.")
    pooled = np.concatenate([np.asarray(chain, dtype=float) for chain in trajectories], axis=0)
    # Pick a high-alpha candidate only from the upper 5% of the prior pilot,
    # then retain the highest-density point in that region. This uses prior
    # pilot draws, never hidden parameters.
    cutoff = np.quantile(pooled[:, 0], 0.95)
    candidates = pooled[pooled[:, 0] >= cutoff]
    log_probabilities = np.asarray(lotka_volterra_log_prob(candidates, x_obs, simulator, mask=mask), dtype=float)
    valid = np.isfinite(log_probabilities)
    if not np.any(valid):
        raise RuntimeError("No finite high-tail initialization candidate was found.")
    selected = np.flatnonzero(valid)[np.argmax(log_probabilities[valid])]
    center = candidates[selected]
    return center, {
        "regime": "high-tail", "center_log_theta": center,
        "selection": "highest partial-data log posterior among prior representative-walker alpha upper 5%",
        "candidate_count": int(candidates.shape[0]), "alpha_log_cutoff": float(cutoff),
        "center_log_prob": float(log_probabilities[selected]), "previous_pilot": str(args.previous_pilot),
    }


def _summarize_chain(result: dict[str, Any], seed: int, walker_index: int, elapsed: float) -> tuple[dict[str, Any], np.ndarray]:
    raw = np.asarray(result["chain"], dtype=float).reshape(-1, 4)
    log_chain = np.asarray(result["log_theta_chain"], dtype=float)
    if not 0 <= walker_index < log_chain.shape[1]:
        raise ValueError(f"diagnostic walker {walker_index} is outside [0, {log_chain.shape[1] - 1}].")
    early_raw = np.asarray(result["chain"], dtype=float)
    log_posterior_chain = np.asarray(result["log_posterior_chain"], dtype=float)
    midpoint = early_raw.shape[0] // 2
    flat_early, flat_late = early_raw[:midpoint].reshape(-1, 4), early_raw[midpoint:].reshape(-1, 4)
    return {
        "seed": seed,
        "runtime_seconds": elapsed,
        "acceptance_fraction": float(result["acceptance_fraction"]),
        "acceptance_fraction_by_walker": np.asarray(result["acceptance_fraction_by_walker"]),
        "autocorr_time_heuristic": np.asarray(result["autocorr_time"]),
        "pooled_walker_ess_heuristic": np.asarray(result["emcee_ess"]),
        "ode_failure_count_production": int(result["ode_failure_count"]),
        "map_log_theta": np.asarray(result["map_log_theta"]),
        "map_estimate": np.asarray(result["map_estimate"]),
        "map_log_prob": float(result["map_log_prob"]),
        "posterior_mean": raw.mean(axis=0),
        "posterior_covariance": np.cov(raw, rowvar=False),
        "posterior_quantiles_05_50_95": np.quantile(raw, [0.05, 0.50, 0.95], axis=0),
        "num_raw_draws": int(raw.shape[0]),
        "mcmc_log_prob_evaluation_count": int(result["mcmc_log_prob_evaluation_count"]),
        "walker_center_log_theta": np.asarray(result["walker_center_log_theta"]),
        "initial_state_log_theta": np.asarray(result["initial_state"]),
        "initialization_diagnostics": result["initialization_diagnostics"],
        "final_log_posterior_by_walker": log_posterior_chain[-1],
        "per_walker_log_parameter_movement": np.linalg.norm(
            log_chain[-1] - np.asarray(result["initial_state"]), axis=1
        ),
        "early_late_mean_l2_delta": float(np.linalg.norm(flat_early.mean(axis=0) - flat_late.mean(axis=0))),
        "early_late_max_quantile_abs_delta": float(np.max(np.abs(
            np.quantile(flat_early, [0.05, 0.5, 0.95], axis=0) - np.quantile(flat_late, [0.05, 0.5, 0.95], axis=0)
        ))),
        "diagnostic_walker_index": walker_index,
    }, log_chain[:, walker_index, :]


def _independent_diagnostics(chains: list[np.ndarray]) -> dict[str, Any]:
    """Use one predetermined trajectory per independently seeded ensemble.

    This deliberately does *not* reinterpret interacting walkers as independent
    chains. The output is a conservative diagnostic, not an ESS for pooled
    resampled posterior draws.
    """
    if len(chains) < 2:
        return {"method": "not_available_fewer_than_two_independent_ensembles"}
    # ArviZ currently uses platformdirs' macOS cache path at import time. Patch
    # only that lookup so a batch run never fails after successful MCMC merely
    # because the user's Library cache is unavailable.
    import platformdirs
    cache_dir = Path(os.environ["GAPSBI_ARVIZ_CACHE"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    platformdirs.user_cache_dir = lambda *_args, **_kwargs: str(cache_dir)
    import arviz as az

    idata = az.from_dict(posterior={"log_theta": np.stack(chains, axis=0)})
    return {
        "method": "ArviZ_rank_normalized_from_fixed_walker_of_independent_ensembles",
        "num_independent_ensembles": len(chains),
        "rank_rhat": az.rhat(idata, method="rank")["log_theta"].values,
        "bulk_ess": az.ess(idata, method="bulk")["log_theta"].values,
        "tail_ess": az.ess(idata, method="tail")["log_theta"].values,
    }


def _pairwise_agreement(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for left, right in combinations(summaries, 2):
        rows.append({
            "seeds": [left["seed"], right["seed"]],
            "posterior_mean_l2_delta": float(np.linalg.norm(np.asarray(left["posterior_mean"]) - np.asarray(right["posterior_mean"]))),
            "posterior_covariance_fro_delta": float(np.linalg.norm(
                np.asarray(left["posterior_covariance"]) - np.asarray(right["posterior_covariance"]), ord="fro")),
            "max_quantile_abs_delta": float(np.max(np.abs(
                np.asarray(left["posterior_quantiles_05_50_95"]) - np.asarray(right["posterior_quantiles_05_50_95"])))),
        })
    return rows


def run_case(args: argparse.Namespace, fraction: float, index: int, logger: logging.Logger) -> None:
    tag = f"eps{round(fraction * 100):03d}_ref{index:02d}"
    result_path = args.output_dir / f"lv_mcar_{tag}_pilot.json"
    prior: dict[str, Any] | None = None
    if result_path.exists():
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        if prior.get("status") == "completed":
            if not args.append_extended_run:
                logger.info("Skipping completed %s", tag)
                return
            if args.extended_production_steps <= 0:
                raise ValueError("--append-extended-run requires --extended-production-steps > 0")
        if not args.rerun_incomplete:
            if prior.get("status") != "completed":
                logger.info("Skipping incomplete %s; use --rerun-incomplete to retry", tag)
                return

    observations, _, metadata, _ = load_reference_posteriors_hdf5(args.full_reference)
    simulator = _simulator(metadata)
    generator = (LotkaVolterraTimeBlockMCARMask(fraction, block_size=5, num_populations=2)
                 if args.mechanism == "mcar" else LotkaVolterraLogTotalMNARMask(fraction))
    mask = generator.generate(observations["x_full"], observations["theta_true"],
                              np.random.default_rng(args.reference_mask_seed))[index]
    x_obs = observations["x_full"][index] * mask
    if prior is not None and prior.get("status") == "completed" and args.append_extended_run:
        # Append-only repair path.  It never overwrites or reruns successful
        # primary ensembles; this exists for correcting a requested stability
        # setting after those primary results have been persisted.
        extended_burn = args.extended_burn_in_steps if args.extended_burn_in_steps is not None else args.burn_in_steps
        result = prior
        result["status"] = "running"
        result.setdefault("extended_ensembles", [])
        attempt = len(result["extended_ensembles"])
        seed = _seed(args.posterior_seed, fraction, index, attempt + 1, extended=True)
        logger.info("Appending extended run for %s attempt=%d burn=%d production=%d", tag, attempt,
                    extended_burn, args.extended_production_steps)
        _write_json_atomic(result_path, result)
        try:
            start = time.perf_counter()
            sampled = run_lotka_volterra_emcee_reference(
                x_obs, simulator=simulator, num_walkers=args.num_walkers,
                burn_in_steps=extended_burn, production_steps=args.extended_production_steps,
                initial_scale=args.initial_scale, mask=mask, rng=np.random.default_rng(seed),
            )
            summary, _ = _summarize_chain(sampled, seed, args.diagnostic_walker_index, time.perf_counter() - start)
            result["extended_ensembles"].append({
                "sampler_settings": {"num_walkers": args.num_walkers, "burn_in_steps": extended_burn,
                                     "production_steps": args.extended_production_steps,
                                     "initial_scale": args.initial_scale},
                "summary": summary,
            })
            result["status"] = "completed"
            logger.info("Completed appended extended run for %s", tag)
        except Exception as error:
            result["status"] = "completed"  # Primary results remain complete.
            result.setdefault("errors", []).append({"type": type(error).__name__, "message": str(error),
                                                     "stage": "append_extended_run"})
            logger.exception("Failed appended extended run for %s", tag)
        finally:
            _write_json_atomic(result_path, result)
        return
    result: dict[str, Any] = {
        "schema": "gapsbi_lv_mcar_pilot_v1", "status": "running", "case": {"fraction": fraction, "reference_index": index},
        "provenance": {"full_reference": str(args.full_reference), "reference_mask_seed": args.reference_mask_seed,
                       "x_full_sha256": _digest(observations["x_full"][index]), "mask_sha256": _digest(mask),
                       "x_obs_sha256": _digest(x_obs), "mask_metadata": generator.metadata()},
        "sampler_settings": {"num_ensembles": args.num_ensembles, "num_walkers": args.num_walkers,
                             "burn_in_steps": args.burn_in_steps, "production_steps": args.production_steps,
                             "initial_scale": args.initial_scale, "diagnostic_walker_index": args.diagnostic_walker_index,
                             "extended_production_steps": args.extended_production_steps,
                             "extended_burn_in_steps": args.extended_burn_in_steps},
        "missing_feature_count": int(np.sum(mask == 0)), "missing_time_count": int(np.sum(mask.reshape(-1, 2)[:, 0] == 0)),
        "initial_observation_masked": bool(mask[0] == 0), "ensembles": [],
        "independent_representative_log_theta_chains": [], "errors": [],
    }
    if args.num_ensembles == 1:
        center, initialization = _targeted_initialization(args, simulator, x_obs, mask)
        result["targeted_initialization"] = initialization
    else:
        center = None
    _write_json_atomic(result_path, result)
    logger.info("Starting %s", tag)
    chains: list[np.ndarray] = []
    try:
        for ensemble in range(args.num_ensembles):
            seed = _seed(args.posterior_seed, fraction, index, ensemble)
            start = time.perf_counter()
            sampled = run_lotka_volterra_emcee_reference(
                x_obs, simulator=simulator, num_walkers=args.num_walkers,
                burn_in_steps=args.burn_in_steps, production_steps=args.production_steps,
                initial_scale=args.initial_scale, mask=mask, walker_center_log_theta=center,
                screen_initialization=args.screen_initialization, rng=np.random.default_rng(seed),
            )
            summary, chain = _summarize_chain(sampled, seed, args.diagnostic_walker_index, time.perf_counter() - start)
            initial_log_probabilities = np.asarray(lotka_volterra_log_prob(
                np.asarray(sampled["initial_state"]), x_obs, simulator, mask=mask
            ), dtype=float)
            if not np.all(np.isfinite(initial_log_probabilities)):
                raise RuntimeError("Initialization produced non-finite partial-data posterior values.")
            summary["initial_log_prob_min_max"] = [float(initial_log_probabilities.min()), float(initial_log_probabilities.max())]
            result["ensembles"].append(summary); chains.append(chain)
            # These are only the fixed representative trajectories used for
            # independent-run diagnostics, not posterior reference samples.
            result["independent_representative_log_theta_chains"].append(chain)
            if args.store_chains:
                chain_path = args.output_dir / f"lv_mcar_{tag}_ensemble{ensemble:02d}_chains.npz"
                _write_npz_atomic(chain_path, log_theta_chain=np.asarray(sampled["log_theta_chain"]),
                                  log_posterior_chain=np.asarray(sampled["log_posterior_chain"]))
                summary["chain_file"] = str(chain_path)
            _write_json_atomic(result_path, result)
            logger.info("Completed %s ensemble=%d acceptance=%.3f", tag, ensemble, summary["acceptance_fraction"])
        result["independent_ensemble_diagnostics"] = _independent_diagnostics(chains)
        result["pairwise_independent_ensemble_agreement"] = _pairwise_agreement(result["ensembles"])
        if args.extended_production_steps:
            seed = _seed(args.posterior_seed, fraction, index, 0, extended=True)
            start = time.perf_counter()
            sampled = run_lotka_volterra_emcee_reference(
                x_obs, simulator=simulator, num_walkers=args.num_walkers,
                burn_in_steps=(args.extended_burn_in_steps if args.extended_burn_in_steps is not None else args.burn_in_steps),
                initial_scale=args.initial_scale, mask=mask, rng=np.random.default_rng(seed),
            )
            result["extended_ensemble"], _ = _summarize_chain(sampled, seed, args.diagnostic_walker_index, time.perf_counter() - start)
        result["status"] = "completed"
        logger.info("Completed %s", tag)
    except Exception as error:
        result["status"] = "failed"
        result["errors"].append({"type": type(error).__name__, "message": str(error)})
        logger.exception("Failed %s", tag)
    finally:
        _write_json_atomic(result_path, result)


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # ArviZ otherwise attempts to write to a user cache that may be unavailable
    # in batch/sandbox contexts.
    os.environ["GAPSBI_ARVIZ_CACHE"] = str(args.output_dir / ".arviz_cache")
    logging.basicConfig(filename=args.output_dir / "lv_mcar_pilot.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("lv_mcar_pilot")
    for fraction, index in args.case:
        run_case(args, fraction, index, logger)


if __name__ == "__main__":
    main()
