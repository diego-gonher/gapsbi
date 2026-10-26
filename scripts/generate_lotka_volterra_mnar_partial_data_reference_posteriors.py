"""Resumable LV/MNAR references; 50% records rather than aborts on IS failure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gapsbi.masks import LotkaVolterraLogTotalMNARMask
from gapsbi.references import (
    DEFAULT_PARTIAL_DATA_REFERENCE_VERSION,
    default_full_reference_path,
    default_partial_reference_path,
    find_lotka_volterra_map,
    importance_reweight_lotka_volterra_mnar,
    load_reference_posteriors_hdf5,
    run_lotka_volterra_emcee_reference,
    save_partial_data_reference_posteriors_hdf5,
)
from gapsbi.simulators import LotkaVolterraSimulator


def encode(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=encode), encoding="utf-8")
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def simulator(metadata: dict) -> LotkaVolterraSimulator:
    s = metadata["simulator"]
    return LotkaVolterraSimulator(
        num_timepoints=s["num_timepoints"], days=s["days"],
        observation_noise_scale=s["observation_noise_scale"], initial_state=tuple(s["initial_state"]),
        prior_log_mean=np.asarray(s["prior_log_mean"]), prior_log_std=np.asarray(s["prior_log_std"]),
        max_state=s["max_state"], ode_rtol=s["ode_rtol"], ode_atol=s["ode_atol"],
    )


def seed(*values: int) -> int:
    return int(np.random.SeedSequence(values).generate_state(1)[0])


def completed_samples(directory: Path) -> np.ndarray:
    """Reconstruct exact saved draws, without resampling an estimator or MCMC chain."""
    final_path = directory / "final_samples.npz"
    if final_path.exists():
        return np.load(final_path)["theta_samples"]
    importance = np.load(directory / "importance.npz")
    status = json.loads((directory / "status.json").read_text())
    indices = np.asarray(status["diagnostics"]["resampling_indices"], dtype=int)
    samples = np.exp(importance["log_theta_proposals"])[indices]
    atomic_npz(final_path, theta_samples=samples, resampling_indices=indices)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--missing-fraction", type=float, required=True, choices=(.1, .25, .5))
    parser.add_argument("--full-reference", type=Path, default=default_full_reference_path("lotka_volterra"))
    parser.add_argument("--reference-version", default=DEFAULT_PARTIAL_DATA_REFERENCE_VERSION)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/reference_generation/lv_mnar"))
    parser.add_argument("--posterior-seed", "--seed", dest="posterior_seed", type=int, default=77123)
    parser.add_argument("--reference-mask-seed", type=int, default=987654)
    parser.add_argument("--num-reference-samples", type=int, default=10_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    sparse = args.missing_fraction == .5
    observations, _, metadata, _ = load_reference_posteriors_hdf5(args.full_reference)
    sim = simulator(metadata)
    generator = LotkaVolterraLogTotalMNARMask(args.missing_fraction)
    mask = generator.generate(observations["x_full"], observations["theta_true"], np.random.default_rng(args.reference_mask_seed))
    x_obs = observations["x_full"] * mask
    tag = f"eps{round(100 * args.missing_fraction):03d}"
    work = args.work_dir / tag
    output = args.output or default_partial_reference_path(
        "lotka_volterra", "mnar", args.missing_fraction, reference_version=args.reference_version,
    )
    successful: list[int] = []
    failed: list[dict] = []
    samples: list[np.ndarray] = []
    diagnostics: list[dict] = []

    for index in range(10):
        directory = work / f"reference_{index:02d}"
        status_path = directory / "status.json"
        prior = json.loads(status_path.read_text()) if status_path.exists() else {}
        if prior.get("status") == "failed":
            failed.append(prior)
            continue
        if prior.get("status") == "completed":
            final = completed_samples(directory)
            if final.shape != (args.num_reference_samples, 4) or not np.all(np.isfinite(final)) or not np.all(final > 0):
                raise ValueError(f"Persisted successful reference {index} is invalid.")
            successful.append(index); samples.append(final); diagnostics.append(prior["diagnostics"])
            continue

        atomic_json(status_path, {"status": "running", "mechanism": "mnar", "reference_index": index})
        chains: list[np.ndarray] = []
        summaries: list[dict] = []
        importance_context: dict = {"latent_completions": 16}
        try:
            # A MAP is only needed if an ensemble is absent; saved observed-data
            # proposal chains are reused unchanged.
            missing_ensemble = any(not (directory / f"ensemble_{ensemble}.npz").exists() for ensemble in range(2))
            map_z = map_log_prob = None
            if missing_ensemble:
                map_z, map_log_prob = find_lotka_volterra_map(x_obs[index], sim, mask=mask[index])
            for ensemble in range(2):
                chain_path = directory / f"ensemble_{ensemble}.npz"
                summary_path = directory / f"ensemble_{ensemble}.json"
                if chain_path.exists() and summary_path.exists():
                    chains.append(np.load(chain_path)["log_theta_chain"])
                    summaries.append(json.loads(summary_path.read_text()))
                    continue
                result = run_lotka_volterra_emcee_reference(
                    x_obs[index], simulator=sim, num_walkers=64, burn_in_steps=1500,
                    production_steps=6000, initial_scale=.3, mask=mask[index],
                    map_log_theta=map_z, map_log_prob=map_log_prob, screen_initialization=True,
                    rng=np.random.default_rng(seed(args.posterior_seed, round(100 * args.missing_fraction), index, ensemble)),
                )
                chain = np.asarray(result["log_theta_chain"])
                atomic_npz(chain_path, log_theta_chain=chain, log_posterior_chain=result["log_posterior_chain"])
                summary = {"acceptance_fraction": result["acceptance_fraction"],
                           "autocorr_time": result["autocorr_time"], "ess": result["emcee_ess"],
                           "ode_failures": result["ode_failure_count"],
                           "initialization": result["initialization_diagnostics"]}
                atomic_json(summary_path, summary)
                chains.append(chain); summaries.append(summary)

            proposals = np.concatenate([
                chain[::max(1, int(np.ceil(np.nanmax(summary["autocorr_time"]))))].reshape(-1, 4)
                for chain, summary in zip(chains, summaries)
            ])
            importance_context["proposal_count"] = int(len(proposals))
            theta, importance = importance_reweight_lotka_volterra_mnar(
                proposals, x_obs[index], mask[index], simulator=sim,
                missing_fraction=args.missing_fraction, latent_completions=16,
                rng=np.random.default_rng(seed(args.posterior_seed, 99, index)),
            )
            weights = importance.pop("parameter_weights")
            importance_context.update(importance)
            atomic_npz(directory / "importance.npz", log_theta_proposals=proposals, parameter_weights=weights)
            selection = np.searchsorted(np.cumsum(weights), (
                np.random.default_rng(seed(args.posterior_seed, 88, index)).random(args.num_reference_samples)
                + np.arange(args.num_reference_samples)
            ) / args.num_reference_samples)
            final = theta[selection]
            min_mcmc_ess = min(float(np.min(summary["ess"])) for summary in summaries)
            if min_mcmc_ess < 1000 or importance["importance_ess"] < 1000 or not np.all(np.isfinite(final)) or not np.all(final > 0) or any(summary["ode_failures"] for summary in summaries):
                raise RuntimeError("quality_gate_failed")
            record = {"mcmc": summaries, "importance": importance, "resampling_indices": selection}
            atomic_npz(directory / "final_samples.npz", theta_samples=final, resampling_indices=selection)
            atomic_json(status_path, {"status": "completed", "mechanism": "mnar", "reference_index": index, "diagnostics": record})
            successful.append(index); samples.append(final); diagnostics.append(record)
        except Exception as error:
            # 50% is deliberately sparse-aware; lower fractions retain the
            # original fail-fast behaviour to avoid changing validated runs.
            record = {"status": "failed", "mechanism": "mnar", "reference_index": index,
                      "reason": type(error).__name__, "message": str(error),
                      "available_ensembles": len(chains), "mcmc": summaries,
                      "importance_sampling": importance_context}
            atomic_json(status_path, record)
            failed.append(record)
            if not sparse:
                raise

    if not successful:
        raise RuntimeError("No LV/MNAR references passed validation; no sparse artifact was written.")
    selected = np.asarray(successful, dtype=np.int64)
    sample_array = np.stack(samples)
    selected_observations = {key: np.asarray(observations[key])[selected] for key in ("theta_true", "x_full")}
    selected_observations.update({"x_obs": x_obs[selected], "mask": mask[selected], "reference_indices": selected})
    metadata_out = {"problem": "lotka_volterra", "missingness_mechanism": "mnar",
        "posterior_method": "screened_observed_data_emcee_parameter_level_importance",
        "posterior_conditioning": "theta_given_x_obs_mask", "mask": generator.metadata(),
        "reference_mask_seed": args.reference_mask_seed, "posterior_seed": args.posterior_seed,
        "full_reference_path": str(args.full_reference), "sampler_settings": {"ensembles": 2, "walkers": 64, "burn_in": 1500, "production": 6000, "latent_completions": 16},
        "chain_work_dir": str(work), "simulator": sim.metadata(), "theta_scaled": False,
        "x_scaled": False, "original_reference_indices": successful, "failed_reference_indices": [item["reference_index"] for item in failed], "sparse_reference_artifact": sparse}
    save_partial_data_reference_posteriors_hdf5(
        output, observations=selected_observations, theta_samples=sample_array, metadata=metadata_out,
        diagnostics={"reference_indices": selected,
            "importance_ess": np.asarray([item["importance"]["importance_ess"] for item in diagnostics]),
            "importance_ess_fraction": np.asarray([item["importance"]["importance_ess_fraction"] for item in diagnostics]),
            "max_weight": np.asarray([item["importance"]["max_normalized_weight"] for item in diagnostics]),
            "zero_weight_fraction": np.asarray([item["importance"]["zero_weight_fraction"] for item in diagnostics])},
        reference_version=args.reference_version, overwrite=args.overwrite,
    )
    manifest = {"status": "completed", "mechanism": "mnar", "missing_fraction": args.missing_fraction,
                "output": str(output), "successful_reference_indices": successful,
                "failed_references": failed, "attempted_reference_indices": list(range(10))}
    atomic_json(work / "campaign_manifest.json", manifest)
    print(output)


if __name__ == "__main__":
    main()
