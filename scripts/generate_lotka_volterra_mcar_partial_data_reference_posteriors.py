"""Resumably generate screened-emcee LV/MCAR partial-data references.

Each independent ensemble is persisted before any reference-level selection is
performed.  Consequently an interrupted campaign resumes from completed
ensembles and never relies on terminal output as a scientific record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from gapsbi.masks import LotkaVolterraTimeBlockMCARMask
from gapsbi.references import (
    DEFAULT_PARTIAL_DATA_REFERENCE_VERSION,
    default_full_reference_path,
    default_partial_reference_path,
    find_lotka_volterra_map,
    load_partial_data_reference_posteriors_hdf5,
    load_reference_posteriors_hdf5,
    run_lotka_volterra_emcee_reference,
    save_partial_data_reference_posteriors_hdf5,
)
from gapsbi.simulators import LotkaVolterraSimulator


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=_json_default), encoding="utf-8")
    temporary.replace(path)


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _hash(array: np.ndarray) -> str:
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


def _seed(base: int, fraction: float, observation: int, ensemble: int, purpose: int) -> int:
    return int(np.random.SeedSequence([base, round(100 * fraction), observation, ensemble, purpose]).generate_state(1)[0])


def _summary(log_chain: np.ndarray, log_prob: np.ndarray, result: dict[str, Any], elapsed: float) -> dict[str, Any]:
    raw = np.exp(log_chain).reshape(-1, 4)
    midpoint = log_chain.shape[0] // 2
    early, late = np.exp(log_chain[:midpoint]).reshape(-1, 4), np.exp(log_chain[midpoint:]).reshape(-1, 4)
    lp_delta = float(result["map_log_prob"]) - log_prob
    return {
        "runtime_seconds": elapsed,
        "acceptance_fraction": float(result["acceptance_fraction"]),
        "acceptance_fraction_by_walker": np.asarray(result["acceptance_fraction_by_walker"]),
        "autocorr_time_heuristic": np.asarray(result["autocorr_time"]),
        "pooled_walker_ess_heuristic": np.asarray(result["emcee_ess"]),
        "ode_failure_count_production": int(result["ode_failure_count"]),
        "mcmc_log_prob_evaluation_count": int(result["mcmc_log_prob_evaluation_count"]),
        "posterior_mean": raw.mean(axis=0),
        "posterior_covariance": np.cov(raw, rowvar=False),
        "posterior_quantiles_05_50_95": np.quantile(raw, [0.05, .5, .95], axis=0),
        "early_late_mean_standardized_delta": np.abs(early.mean(0) - late.mean(0)) / np.maximum(raw.std(0, ddof=1), 1e-12),
        "early_late_quantile_iqr_relative_delta": np.max(np.abs(
            np.quantile(early, [.05, .5, .95], axis=0) - np.quantile(late, [.05, .5, .95], axis=0)
        ), axis=0) / np.maximum(np.quantile(raw, .75, axis=0) - np.quantile(raw, .25, axis=0), 1e-12),
        "low_density_fraction_by_walker": np.mean(lp_delta > 20.0, axis=0),
        "final_log_posterior_by_walker": log_prob[-1],
        "per_walker_log_parameter_movement": np.linalg.norm(log_chain[-1] - np.asarray(result["initial_state"]), axis=1),
        "map_log_theta": np.asarray(result["map_log_theta"]),
        "map_log_prob": float(result["map_log_prob"]),
        "initialization_diagnostics": result["initialization_diagnostics"],
        "initial_log_theta": np.asarray(result["initial_state"]),
        "final_log_theta": log_chain[-1],
    }


def _arviz_independent(chains: list[np.ndarray]) -> dict[str, Any]:
    """Diagnostic from a predetermined walker of each independent ensemble.

    This intentionally does not mistake interacting walkers for independent
    chains and is supplementary to the full-ensemble agreement checks.
    """
    import platformdirs
    cache = Path(os.environ.get("GAPSBI_ARVIZ_CACHE", "/tmp/gapsbi_arviz_cache"))
    cache.mkdir(parents=True, exist_ok=True)
    platformdirs.user_cache_dir = lambda *_args, **_kwargs: str(cache)
    import arviz as az
    idata = az.from_dict(posterior={"log_theta": np.stack([c[:, 0, :] for c in chains])})
    return {
        "method": "ArviZ rank-normalized diagnostic from walker 0 of independent ensembles; supplementary only",
        "rank_rhat": az.rhat(idata, method="rank")["log_theta"].values,
        "bulk_ess": az.ess(idata, method="bulk")["log_theta"].values,
        "tail_ess": az.ess(idata, method="tail")["log_theta"].values,
    }


def _quality(ensembles: list[dict[str, Any]]) -> dict[str, Any]:
    means = [np.asarray(x["posterior_mean"]) for x in ensembles]
    covs = [np.asarray(x["posterior_covariance"]) for x in ensembles]
    qs = [np.asarray(x["posterior_quantiles_05_50_95"]) for x in ensembles]
    standard_deviation = np.sqrt(np.maximum(np.diag((covs[0] + covs[1]) / 2), 1e-16))
    iqr = np.maximum((qs[0][2] - qs[0][0] + qs[1][2] - qs[1][0]) / 2, 1e-16)
    mean_delta = np.abs(means[0] - means[1]) / standard_deviation
    quantile_delta = np.max(np.abs(qs[0] - qs[1]), axis=0) / iqr
    covariance_relative_fro = float(np.linalg.norm(covs[0] - covs[1], ord="fro") / max(np.linalg.norm((covs[0] + covs[1]) / 2, ord="fro"), 1e-16))
    violations: list[str] = []
    for number, item in enumerate(ensembles):
        if not np.all(np.isfinite(item["autocorr_time_heuristic"])) or np.min(item["pooled_walker_ess_heuristic"]) < 1000:
            violations.append(f"ensemble_{number}: heuristic_ess_below_1000_or_unavailable")
        if int(item["ode_failure_count_production"]) != 0:
            violations.append(f"ensemble_{number}: ODE_failures")
        if np.max(item["low_density_fraction_by_walker"]) > .01:
            violations.append(f"ensemble_{number}: persistent_low_density_walker")
        if np.max(item["early_late_mean_standardized_delta"]) > .10 or np.max(item["early_late_quantile_iqr_relative_delta"]) > .10:
            violations.append(f"ensemble_{number}: early_late_instability")
    if np.max(mean_delta) > .10:
        violations.append("cross_ensemble_standardized_mean_delta_above_0.10")
    if np.max(quantile_delta) > .10:
        violations.append("cross_ensemble_quantile_iqr_delta_above_0.10")
    if covariance_relative_fro > .10:
        violations.append("cross_ensemble_covariance_relative_fro_above_0.10")
    return {"passed": not violations, "violations": violations,
            "cross_ensemble_standardized_mean_delta": mean_delta,
            "cross_ensemble_quantile_iqr_relative_delta": quantile_delta,
            "cross_ensemble_covariance_relative_fro_delta": covariance_relative_fro}


def _select(chains: list[np.ndarray], taus: list[np.ndarray], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, list[int]]:
    thin = [max(1, int(np.ceil(np.nanmax(tau)))) for tau in taus]
    pools: list[tuple[np.ndarray, np.ndarray]] = []
    for ensemble, (chain, stride) in enumerate(zip(chains, thin, strict=True)):
        positions = chain[::stride]
        values = np.exp(positions).reshape(-1, 4)
        iterations, walkers = np.indices(positions.shape[:2])
        source = np.column_stack((np.full(values.shape[0], ensemble), (iterations * stride).reshape(-1), walkers.reshape(-1)))
        if values.shape[0] < 5000:
            raise RuntimeError(f"ensemble {ensemble} thinned pool has only {values.shape[0]} < 5000 samples; extend it before selection")
        pools.append((values, source))
    selected_values, selected_sources = [], []
    for values, source in pools:
        indices = rng.choice(values.shape[0], size=5000, replace=False)
        selected_values.append(values[indices]); selected_sources.append(source[indices])
    return np.concatenate(selected_values), np.concatenate(selected_sources), thin


def _run_observation(args: argparse.Namespace, simulator: LotkaVolterraSimulator, x_obs: np.ndarray, mask: np.ndarray,
                     fraction: float, index: int, work: Path) -> tuple[np.ndarray, dict[str, Any]]:
    status_path, selection_path = work / "status.json", work / "selection.npz"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"status": "pending", "ensembles": {}}
    status.update({"status": "running", "fraction": fraction, "observation_index": index})
    _write_json_atomic(status_path, status)
    map_seed = _seed(args.posterior_seed, fraction, index, 0, 1)
    map_z, map_lp = find_lotka_volterra_map(x_obs, simulator, mask=mask)
    status["map_log_theta"], status["map_log_prob"], status["map_seed"] = map_z, map_lp, map_seed
    _write_json_atomic(status_path, status)
    ensemble_summaries, chains, taus = [], [], []
    for ensemble in range(2):
        folder = work / f"ensemble_{ensemble:02d}"; json_path, npz_path = folder / "result.json", folder / "chain.npz"
        if json_path.exists() and npz_path.exists():
            saved = json.loads(json_path.read_text())
            with np.load(npz_path) as data:
                chain, log_prob = data["log_theta_chain"], data["log_posterior_chain"]
            if saved.get("status") == "completed" and chain.shape == (args.production_steps, args.num_walkers, 4):
                ensemble_summaries.append(saved["summary"]); chains.append(chain); taus.append(np.asarray(saved["summary"]["autocorr_time_heuristic"])); continue
        seed = _seed(args.posterior_seed, fraction, index, ensemble, 2)
        _write_json_atomic(json_path, {"status": "running", "seed": seed})
        began = time.perf_counter()
        try:
            result = run_lotka_volterra_emcee_reference(
                x_obs, simulator=simulator, num_walkers=args.num_walkers, burn_in_steps=args.burn_in_steps,
                production_steps=args.production_steps, initial_scale=args.initial_scale, mask=mask,
                map_log_theta=map_z, map_log_prob=map_lp, screen_initialization=True, rng=np.random.default_rng(seed),
            )
            chain, log_prob = np.asarray(result["log_theta_chain"]), np.asarray(result["log_posterior_chain"])
            summary = _summary(chain, log_prob, result, time.perf_counter() - began)
            _write_npz_atomic(npz_path, log_theta_chain=chain, log_posterior_chain=log_prob,
                              initial_log_theta=np.asarray(result["initial_state"]), final_log_theta=chain[-1])
            _write_json_atomic(json_path, {"status": "completed", "seed": seed, "summary": summary})
        except Exception as error:
            _write_json_atomic(json_path, {"status": "failed", "seed": seed, "error": repr(error)})
            raise
        ensemble_summaries.append(summary); chains.append(chain); taus.append(np.asarray(summary["autocorr_time_heuristic"]))
    quality = _quality(ensemble_summaries)
    independent = _arviz_independent(chains)
    if not quality["passed"]:
        status.update({"status": "quality_failed", "quality": quality, "independent_diagnostics": independent, "ensembles": ensemble_summaries})
        _write_json_atomic(status_path, status)
        raise RuntimeError(f"Quality gates failed for eps{fraction:.2f} observation {index}: {quality['violations']}")
    samples, sources, thin = _select(chains, taus, np.random.default_rng(_seed(args.posterior_seed, fraction, index, 0, 3)))
    if not np.all(np.isfinite(samples)) or not np.all(samples > 0):
        raise RuntimeError("Selected LV posterior samples are nonfinite or nonpositive.")
    _write_npz_atomic(selection_path, theta_samples=samples, source_indices=sources, thinning=np.asarray(thin, dtype=np.int64))
    status.update({"status": "completed", "quality": quality, "independent_diagnostics": independent, "ensembles": ensemble_summaries,
                   "selection": {"selection_seed": _seed(args.posterior_seed, fraction, index, 0, 3), "thinning": thin,
                                 "source_chain_path": str(work), "equal_samples_per_ensemble": 5000}})
    _write_json_atomic(status_path, status)
    return samples, status


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--missing-fraction", type=float, required=True, choices=(.10, .25, .50))
    p.add_argument("--full-reference", type=Path, default=default_full_reference_path("lotka_volterra"))
    p.add_argument("--reference-version", default=DEFAULT_PARTIAL_DATA_REFERENCE_VERSION)
    p.add_argument("--output", type=Path)
    p.add_argument("--work-dir", type=Path, default=Path("outputs/reference_generation/lv_mcar"))
    p.add_argument("--reference-mask-seed", type=int, default=987654); p.add_argument("--posterior-seed", type=int, default=901245)
    p.add_argument("--num-walkers", type=int, default=64); p.add_argument("--burn-in-steps", type=int, default=1500)
    p.add_argument("--production-steps", type=int, default=6000); p.add_argument("--initial-scale", type=float, default=.30)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    observations, _, metadata, _ = load_reference_posteriors_hdf5(args.full_reference)
    simulator = _simulator(metadata)
    generator = LotkaVolterraTimeBlockMCARMask(args.missing_fraction, block_size=5, num_populations=2)
    mask = generator.generate(observations["x_full"], observations["theta_true"], np.random.default_rng(args.reference_mask_seed))
    x_obs = observations["x_full"] * mask
    tag = f"eps{round(100 * args.missing_fraction):03d}"
    work = args.work_dir / tag; work.mkdir(parents=True, exist_ok=True)
    output = args.output or default_partial_reference_path("lotka_volterra", "mcar", args.missing_fraction, reference_version=args.reference_version)
    selected = [] ; statuses = []
    for index in range(10):
        print(f"LV/MCAR {tag}, reference {index + 1}/10", flush=True)
        samples, status = _run_observation(args, simulator, x_obs[index], mask[index], args.missing_fraction, index, work / f"reference_{index:02d}")
        selected.append(samples); statuses.append(status)
    theta_samples = np.stack(selected)
    artifact_observations = {"theta_true": observations["theta_true"], "x_full": observations["x_full"], "x_obs": x_obs, "mask": mask}
    diagnostics = {
        "posterior_sample_mean": theta_samples.mean(1), "posterior_sample_std": theta_samples.std(1),
        "acceptance_fraction": np.asarray([np.mean([e["acceptance_fraction"] for e in s["ensembles"]]) for s in statuses]),
        "min_heuristic_ess": np.asarray([min(np.min(e["pooled_walker_ess_heuristic"]) for e in s["ensembles"]) for s in statuses]),
        "source_indices": np.stack([np.load(work / f"reference_{i:02d}" / "selection.npz")["source_indices"] for i in range(10)]),
        "thinning": np.asarray([s["selection"]["thinning"] for s in statuses], dtype=np.int64),
        "num_nonfinite_posterior_samples": np.asarray(np.count_nonzero(~np.isfinite(theta_samples)), dtype=np.int64),
    }
    artifact_metadata = {"problem": "lotka_volterra", "missingness_mechanism": "mcar", "posterior_conditioning": "theta_given_x_obs_mask",
                         "posterior_method": "two_independent_screened_map_hessian_emcee_ensembles", "mask": generator.metadata(),
                         "reference_mask_seed": args.reference_mask_seed, "posterior_seed": args.posterior_seed, "simulator": simulator.metadata(),
                         "theta_scaled": False, "x_scaled": False, "parameterization": "MCMC_log_theta_saved_raw_theta",
                         "full_reference_path": str(args.full_reference), "full_x_sha256": _hash(observations["x_full"]),
                         "theta_true_sha256": _hash(observations["theta_true"]), "chain_work_dir": str(work),
                         "sampler_settings": {"ensembles": 2, "walkers": args.num_walkers, "burn_in_steps": args.burn_in_steps, "production_steps": args.production_steps, "screen_log_density_drop": 16.68842079082944}}
    save_partial_data_reference_posteriors_hdf5(output, observations=artifact_observations, theta_samples=theta_samples,
        metadata=artifact_metadata, diagnostics=diagnostics, reference_version=args.reference_version, overwrite=args.overwrite)
    loaded_obs, loaded_samples, _, _ = load_partial_data_reference_posteriors_hdf5(output)
    if not (np.array_equal(loaded_obs["x_full"], observations["x_full"]) and np.array_equal(loaded_obs["mask"], mask)
            and np.array_equal(loaded_obs["x_obs"], x_obs) and np.array_equal(loaded_samples, theta_samples)):
        raise RuntimeError("HDF5 round-trip validation failed.")
    _write_json_atomic(work / "campaign_manifest.json", {"status": "completed", "output": str(output), "fraction": args.missing_fraction,
        "num_references": 10, "all_quality_gates_passed": True, "x_full_sha256": _hash(observations["x_full"]),
        "mask_sha256": _hash(mask), "reference_status_paths": [str(work / f"reference_{i:02d}" / "status.json") for i in range(10)]})
    print(f"Saved and validated: {output}", flush=True)


if __name__ == "__main__":
    main()
