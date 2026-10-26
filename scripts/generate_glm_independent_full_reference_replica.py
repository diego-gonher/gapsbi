#!/usr/bin/env python3
"""Generate a resumable independent MCMC replica of the GLM full references.

This deliberately loads the immutable canonical observations rather than
recreating them from an observation seed.  Per-observation checkpoints make a
stopped run safe to resume without reusing a completed chain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import emcee
import h5py
import numpy as np
import scipy

from gapsbi.references import (
    load_reference_posteriors_hdf5,
    run_glm_emcee_reference,
    save_reference_posteriors_hdf5,
)
from gapsbi.simulators import GLMSimulator


CANONICAL = Path("references/reference_posteriors_v1/glm_references.h5")
DEFAULT_OUTPUT = Path("references/reference_posteriors_v1/glm_references_independent_replica_v1.h5")
DEFAULT_WORK = Path("outputs/reference_generation/glm_full_independent_replica_v1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.view(np.uint8)).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
        os.replace(temp_name, path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.stem + ".", suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def seed_map(master_seed: int, count: int) -> list[dict[str, int]]:
    """Create independent initialization, emcee, and selection streams."""
    output = []
    for child in np.random.SeedSequence(master_seed).spawn(count):
        init, sampler, selection = child.spawn(3)
        output.append(
            {
                # emcee 3.x stores an MT19937 RandomState and therefore
                # requires a legacy 32-bit seed.
                "initialization": int(init.generate_state(1, dtype=np.uint32)[0]),
                "emcee": int(sampler.generate_state(1, dtype=np.uint32)[0]),
                "selection": int(selection.generate_state(1, dtype=np.uint32)[0]),
            }
        )
    return output


def run_one(
    x_full: np.ndarray,
    simulator: GLMSimulator,
    settings: dict[str, int | float],
    seeds: dict[str, int],
) -> dict[str, np.ndarray | float]:
    result = run_glm_emcee_reference(
        x_full,
        simulator=simulator,
        num_walkers=int(settings["num_walkers"]),
        burn_in_steps=int(settings["burn_in_steps"]),
        production_steps=int(settings["production_steps"]),
        initial_scale=float(settings["initial_scale"]),
        rng=np.random.default_rng(seeds["initialization"]),
        emcee_seed=seeds["emcee"],
    )
    chain = np.asarray(result["chain"], dtype=float)
    flat = chain.reshape(-1, simulator.theta_dim)
    selected = np.random.default_rng(seeds["selection"]).choice(
        flat.shape[0], size=int(settings["num_reference_samples"]), replace=False
    )
    result["theta_samples"] = flat[selected]
    result["selection_indices"] = selected.astype(np.int64)
    trace_steps = min(int(settings["trace_num_steps"]), chain.shape[0])
    trace_walkers = min(int(settings["trace_num_walkers"]), chain.shape[1])
    trace_index = np.linspace(0, chain.shape[0] - 1, trace_steps, dtype=int)
    result["mcmc_trace"] = chain[trace_index, :trace_walkers]
    return result


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unavailable"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=CANONICAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--master-seed", type=int, default=684_512_309)
    parser.add_argument("--reference-index", type=int, action="append")
    parser.add_argument("--overwrite", action="store_true", help="Only permit replacing the final output artifact.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; refusing to overwrite it.")
    observations, original_samples, source_metadata, source_diagnostics = load_reference_posteriors_hdf5(args.canonical)
    required = ("num_walkers", "burn_in_steps", "production_steps", "initial_scale", "num_reference_samples")
    settings = {key: source_metadata[key] for key in required}
    settings["trace_num_steps"] = int(source_diagnostics["mcmc_trace"].shape[1])
    settings["trace_num_walkers"] = int(source_diagnostics["mcmc_trace"].shape[2])
    n_ref = original_samples.shape[0]
    if args.reference_index is None:
        requested = list(range(n_ref))
    else:
        requested = sorted(set(args.reference_index))
        if any(index < 0 or index >= n_ref for index in requested):
            raise ValueError(f"reference indices must lie in [0, {n_ref - 1}].")
    simulator_meta = source_metadata["simulator"]
    simulator = GLMSimulator(
        dim=int(simulator_meta["dim"]), duration=int(simulator_meta["duration"]),
        stimulus_seed=int(simulator_meta["stimulus_seed"]), summary="raw",
    )
    seeds = seed_map(args.master_seed, n_ref)
    status_path = args.work_dir / "campaign_manifest.json"
    manifest: dict[str, Any] = {
        "schema": "gapsbi_glm_independent_full_reference_replica_v1",
        "status": "running",
        "canonical_path": str(args.canonical),
        "canonical_sha256": sha256_file(args.canonical),
        "output_path": str(args.output),
        "master_seed": int(args.master_seed),
        "settings": settings,
        "source_observation_sha256": sha256_array(observations["x_full"]),
        "source_theta_true_sha256": sha256_array(observations["theta_true"]),
        "references": {},
    }
    if status_path.is_file():
        manifest = json.loads(status_path.read_text(encoding="utf-8"))
        if manifest["canonical_sha256"] != sha256_file(args.canonical):
            raise RuntimeError("Existing work directory belongs to a different canonical artifact.")
        if manifest["settings"] != settings or manifest["master_seed"] != int(args.master_seed):
            raise RuntimeError("Existing work directory has incompatible settings or master seed.")
    manifest["status"] = "running"
    atomic_json(status_path, manifest)

    for index in requested:
        key = str(index)
        checkpoint = args.work_dir / f"reference_{index:02d}.npz"
        prior = manifest["references"].get(key, {})
        if prior.get("status") == "complete" and checkpoint.is_file():
            continue
        manifest["references"][key] = {"status": "running", "seeds": seeds[index]}
        atomic_json(status_path, manifest)
        try:
            result = run_one(observations["x_full"][index], simulator, settings, seeds[index])
            theta = np.asarray(result["theta_samples"])
            if theta.shape != (int(settings["num_reference_samples"]), simulator.theta_dim) or not np.all(np.isfinite(theta)):
                raise RuntimeError("Produced invalid posterior samples.")
            arrays = {
                "theta_samples": theta,
                "selection_indices": np.asarray(result["selection_indices"]),
                "acceptance_fraction": np.asarray(result["acceptance_fraction"]),
                "autocorr_time": np.asarray(result["autocorr_time"]),
                "emcee_ess": np.asarray(result["emcee_ess"]),
                "split_rhat": np.asarray(result["split_rhat"]),
                "walker_group_mean_l2_max": np.asarray(result["walker_group_mean_l2_max"]),
                "walker_group_std_l2_max": np.asarray(result["walker_group_std_l2_max"]),
                "map_estimate": np.asarray(result["map_estimate"]),
                "map_log_prob": np.asarray(result["map_log_prob"]),
                "mcmc_trace": np.asarray(result["mcmc_trace"]),
            }
            atomic_npz(checkpoint, **arrays)
            manifest["references"][key] = {
                "status": "complete", "seeds": seeds[index], "checkpoint": str(checkpoint),
                "acceptance_fraction": float(result["acceptance_fraction"]),
                "minimum_emcee_ess": float(np.nanmin(result["emcee_ess"])),
            }
        except Exception as error:
            manifest["references"][key] = {"status": "failed", "seeds": seeds[index], "error": repr(error)}
            manifest["status"] = "failed"
            atomic_json(status_path, manifest)
            raise
        atomic_json(status_path, manifest)

    incomplete = [str(i) for i in range(n_ref) if manifest["references"].get(str(i), {}).get("status") != "complete"]
    if incomplete:
        manifest["status"] = "incomplete"
        atomic_json(status_path, manifest)
        print(f"Completed requested references; remaining={incomplete}")
        return

    loaded = [np.load(args.work_dir / f"reference_{i:02d}.npz", allow_pickle=False) for i in range(n_ref)]
    theta = np.stack([entry["theta_samples"] for entry in loaded])
    if np.array_equal(theta, original_samples):
        raise RuntimeError("Independent replica is byte-identical to the canonical posterior samples.")
    diagnostics = {
        "posterior_sample_mean": theta.mean(axis=1),
        "posterior_sample_std": theta.std(axis=1),
        "acceptance_fraction": np.asarray([entry["acceptance_fraction"] for entry in loaded]),
        "autocorr_time": np.stack([entry["autocorr_time"] for entry in loaded]),
        "emcee_ess": np.stack([entry["emcee_ess"] for entry in loaded]),
        "split_rhat": np.stack([entry["split_rhat"] for entry in loaded]),
        "walker_group_mean_l2_max": np.asarray([entry["walker_group_mean_l2_max"] for entry in loaded]),
        "walker_group_std_l2_max": np.asarray([entry["walker_group_std_l2_max"] for entry in loaded]),
        "map_estimate": np.stack([entry["map_estimate"] for entry in loaded]),
        "map_log_prob": np.asarray([entry["map_log_prob"] for entry in loaded]),
        "mcmc_trace": np.stack([entry["mcmc_trace"] for entry in loaded]),
        "selection_indices": np.stack([entry["selection_indices"] for entry in loaded]),
        "original_new_mean_l2": np.linalg.norm(theta.mean(axis=1) - original_samples.mean(axis=1), axis=1),
        "original_new_cov_fro": np.asarray([np.linalg.norm(np.cov(theta[i], rowvar=False) - np.cov(original_samples[i], rowvar=False), ord="fro") for i in range(n_ref)]),
        "original_new_max_abs_quantile_delta": np.asarray([np.max(np.abs(np.quantile(theta[i], [.05, .5, .95], axis=0) - np.quantile(original_samples[i], [.05, .5, .95], axis=0))) for i in range(n_ref)]),
    }
    if not np.all(np.isfinite(diagnostics["emcee_ess"])) or np.nanmin(diagnostics["emcee_ess"]) < 1_000:
        raise RuntimeError("Replica fails the minimum numerical ESS gate (1,000 per parameter).")
    metadata = dict(source_metadata)
    metadata.update({
        "reference_version": "reference_posteriors_v1_independent_replica",
        "posterior_seed": int(args.master_seed),
        "posterior_method": "emcee_ensemble_mcmc_independent_replica",
        "independent_replica": True,
        "source_reference_path": str(args.canonical),
        "source_reference_sha256": sha256_file(args.canonical),
        "source_observation_sha256": sha256_array(observations["x_full"]),
        "source_theta_true_sha256": sha256_array(observations["theta_true"]),
        "original_reference_indices": list(range(n_ref)),
        "master_seed": int(args.master_seed),
        "per_reference_seeds": seeds,
        "emcee_version": emcee.__version__, "numpy_version": np.__version__,
        "scipy_version": scipy.__version__, "python_version": sys.version,
        "platform": platform.platform(), "git_revision": git_revision(),
        "sampler_random_state": "per-reference emcee.RandomState seeded explicitly",
    })
    save_reference_posteriors_hdf5(
        args.output, observations=observations, theta_samples=theta, metadata=metadata,
        diagnostics=diagnostics, reference_version="reference_posteriors_v1_independent_replica",
        overwrite=args.overwrite,
    )
    manifest["status"] = "complete"
    manifest["artifact_sha256"] = sha256_file(args.output)
    manifest["diagnostics"] = {
        "minimum_emcee_ess": float(np.nanmin(diagnostics["emcee_ess"])),
        "acceptance_min": float(np.min(diagnostics["acceptance_fraction"])),
        "acceptance_max": float(np.max(diagnostics["acceptance_fraction"])),
    }
    atomic_json(status_path, manifest)
    print(f"Completed independent GLM full-reference replica: {args.output}")


if __name__ == "__main__":
    main()
