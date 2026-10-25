"""Pilot validation for GLM/MCAR partial-data reference sampling.

This does not write reference artifacts.  It compares independent emcee
ensembles with an exact Laplace-proposal independence-MH implementation on
stored GLM reference observations and their evaluation masks.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import arviz as az
import numpy as np

from gapsbi.masks import PointMCARMask
from gapsbi.references import (
    load_reference_posteriors_hdf5,
    run_glm_observed_emcee_reference,
    run_glm_observed_independence_mh,
)
from gapsbi.simulators import GLMSimulator


def _moments(samples: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": np.mean(samples, axis=0).tolist(),
        "variance": np.var(samples, axis=0, ddof=1).tolist(),
        "quantiles_05_50_95": np.quantile(samples, [0.05, 0.5, 0.95], axis=0).tolist(),
    }


def _arviz_diagnostics(chains: list[np.ndarray]) -> dict[str, list[float]]:
    # Each array is (steps, walkers, theta_dim).  We use the same walker index
    # from independently initialized ensembles as independent chains; walkers
    # within an ensemble are not treated as independent chains.
    independent = np.stack([chain[:, 0, :] for chain in chains])
    idata = az.from_dict(posterior={"theta": independent})
    return {
        "rhat_rank": az.rhat(idata, method="rank").theta.values.tolist(),
        "ess_bulk": az.ess(idata, method="bulk").theta.values.tolist(),
        "ess_tail": az.ess(idata, method="tail").theta.values.tolist(),
    }


def _arviz_independent_chain_diagnostics(chains: list[np.ndarray]) -> dict[str, list[float]]:
    idata = az.from_dict(posterior={"theta": np.stack(chains)})
    return {
        "rhat_rank": az.rhat(idata, method="rank").theta.values.tolist(),
        "ess_bulk": az.ess(idata, method="bulk").theta.values.tolist(),
        "ess_tail": az.ess(idata, method="tail").theta.values.tolist(),
    }


def _compare(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left_moments, right_moments = _moments(left), _moments(right)
    return {
        "mean_l2_delta": float(np.linalg.norm(np.mean(left, axis=0) - np.mean(right, axis=0))),
        "max_abs_mean_delta": float(np.max(np.abs(np.mean(left, axis=0) - np.mean(right, axis=0)))),
        "covariance_fro_delta": float(np.linalg.norm(np.cov(left, rowvar=False) - np.cov(right, rowvar=False), ord="fro")),
        "max_abs_quantile_delta": float(np.max(np.abs(
            np.asarray(left_moments["quantiles_05_50_95"])
            - np.asarray(right_moments["quantiles_05_50_95"])
        ))),
    }


def _run_emcee_ensembles(
    x_obs: np.ndarray, mask: np.ndarray, simulator: GLMSimulator, args: argparse.Namespace, seed: int,
) -> tuple[list[np.ndarray], list[dict], float]:
    chains, diagnostics = [], []
    started = time.perf_counter()
    for ensemble in range(args.num_ensembles):
        result = run_glm_observed_emcee_reference(
            x_obs, mask, simulator=simulator, num_walkers=args.num_walkers,
            burn_in_steps=args.burn_in, production_steps=args.production_steps,
            initial_scale=args.initial_scale, rng=np.random.default_rng(seed + ensemble),
        )
        chains.append(np.asarray(result.pop("chain")))
        diagnostics.append(result)
    return chains, diagnostics, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-reference", type=Path, default=Path("references/reference_posteriors_v1/glm_references.h5"))
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis/glm_mcar_partial_reference_pilot.json"))
    parser.add_argument("--reference-mask-seed", type=int, default=987_654)
    parser.add_argument("--num-ensembles", type=int, default=4)
    parser.add_argument("--num-walkers", type=int, default=64)
    parser.add_argument("--burn-in", type=int, default=750)
    parser.add_argument("--production-steps", type=int, default=2_000)
    parser.add_argument("--initial-scale", type=float, default=0.05)
    parser.add_argument("--mh-steps", type=int, default=30_000)
    parser.add_argument("--mh-burn-in", type=int, default=1_000)
    parser.add_argument("--mh-thin", type=int, default=2)
    parser.add_argument("--seed", type=int, default=71_000)
    args = parser.parse_args()
    observations, full_samples, metadata, _ = load_reference_posteriors_hdf5(args.full_reference)
    simulator_metadata = metadata["simulator"]
    simulator = GLMSimulator(dim=simulator_metadata["dim"], duration=simulator_metadata["duration"],
                             stimulus_seed=simulator_metadata["stimulus_seed"], summary="raw")
    result: dict[str, object] = {
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "cases": {},
    }

    # Include reference zero and the mask with the fewest observed values for
    # each nominal fraction; the latter is the challenging realized case.
    for fraction in (0.10, 0.25, 0.50):
        mask = PointMCARMask(fraction).generate(
            observations["x_full"], observations["theta_true"], np.random.default_rng(args.reference_mask_seed)
        )
        indices = sorted({0, int(np.argmin(mask.sum(axis=1)))})
        for index in indices:
            x_obs, one_mask = observations["x_full"][index] * mask[index], mask[index]
            emcee_chains, emcee_details, emcee_runtime = _run_emcee_ensembles(
                x_obs, one_mask, simulator, args, args.seed + int(fraction * 10_000) + index * 100
            )
            emcee_samples = np.concatenate([chain.reshape(-1, simulator.theta_dim) for chain in emcee_chains])
            mh_chains, mh_details, mh_runtime = [], [], 0.0
            for chain_index in range(args.num_ensembles):
                started = time.perf_counter()
                mh = run_glm_observed_independence_mh(
                    x_obs, one_mask, simulator=simulator, num_steps=args.mh_steps,
                    burn_in_steps=args.mh_burn_in, thin=args.mh_thin,
                    rng=np.random.default_rng(args.seed + 50_000 + int(fraction * 10_000) + index * 100 + chain_index),
                )
                mh_runtime += time.perf_counter() - started
                mh_chains.append(np.asarray(mh.pop("chain")))
                mh_details.append(mh)
            mh_samples = np.concatenate(mh_chains)
            result["cases"][f"eps{int(fraction * 100):03d}_obs{index}"] = {
                "fraction": fraction, "reference_index": index,
                "num_observed": int(one_mask.sum()), "num_missing": int((one_mask == 0).sum()),
                "emcee": {"runtime_sec": emcee_runtime, "diagnostics": _arviz_diagnostics(emcee_chains),
                          "ensemble_details": emcee_details, "moments": _moments(emcee_samples)},
                "independence_mh": {"runtime_sec": mh_runtime,
                                    "acceptance_fraction": [float(d["acceptance_fraction"]) for d in mh_details],
                                    "diagnostics": _arviz_independent_chain_diagnostics(mh_chains),
                                    "moments": _moments(mh_samples)},
                "emcee_vs_independence_mh": _compare(emcee_samples, mh_samples),
            }

    # A full mask targets precisely the original full-reference posterior.
    full_mask = np.ones_like(observations["x_full"][0], dtype=np.int8)
    full_chains, _, full_runtime = _run_emcee_ensembles(
        observations["x_full"][0], full_mask, simulator, args, args.seed + 90_000
    )
    new_full_samples = np.concatenate([chain.reshape(-1, simulator.theta_dim) for chain in full_chains])
    result["full_mask_vs_original_reference_obs0"] = {
        "runtime_sec": full_runtime,
        "comparison": _compare(new_full_samples, full_samples[0]),
        "diagnostics": _arviz_diagnostics(full_chains),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=lambda x: np.asarray(x).tolist()))
    print(f"Saved pilot diagnostics: {args.output}")


if __name__ == "__main__":
    main()
