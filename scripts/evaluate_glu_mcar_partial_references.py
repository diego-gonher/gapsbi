"""Evaluate saved GLU MCAR checkpoints against partial-data references.

This is intentionally limited to the two naive methods already trained in the
benchmark: zero imputation and zero-imputation-plus-mask augmentation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from sbi.neural_nets import posterior_nn

from gapsbi.evaluation.posterior_sampling import sample_posteriors_once
from gapsbi.evaluation.reference_metrics import compute_reference_metrics
from gapsbi.methods.imputation import zero_impute
from gapsbi.methods.mask_augmentation import make_zero_imputed_mask_augmented_input
from gapsbi.methods.sbi_npe import FixedSplitNPE_C
from gapsbi.preprocessing.scalers import make_scaled_theta_prior, transform_x_obs_with_fitted_scaler
from gapsbi.references import load_glu_mcar_partial_data_reference_posteriors_hdf5
from gapsbi.utils.seeding import set_all_seeds


METHODS = {
    "zero": "npe_zero_imputation",
    "zero_mask": "npe_mask_augmentation",
}
BUDGETS = ("low", "mid", "high")
FRACTIONS = (0.10, 0.25, 0.50)
SEEDS = (101, 202, 303, 404, 505)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("."))
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("references/partial_data_references_v1"),
    )
    parser.add_argument("--num-posterior-samples", type=int, default=10_000)
    parser.add_argument("--max-sampling-time", type=float, default=30.0)
    parser.add_argument("--method", choices=tuple(METHODS), action="append")
    parser.add_argument("--budget", choices=BUDGETS, action="append")
    parser.add_argument("--fraction", type=float, action="append")
    return parser.parse_args()


def reference_path(reference_root: Path, fraction: float) -> Path:
    return reference_root / f"glu_mcar_eps{int(round(100 * fraction)):03d}_references.h5"


def checkpoint_path(
    outputs_root: Path, method_dir: str, budget: str, fraction: float, seed: int
) -> Path:
    return (
        outputs_root
        / f"outputs_{budget}_sim_budget"
        / method_dir
        / "glu/mcar"
        / f"glu_mcar_eps{int(round(100 * fraction)):03d}"
        / f"seed_{seed}/model_checkpoint.pt"
    )


def rebuild_posterior(checkpoint: dict[str, Any]) -> Any:
    """Reconstruct the saved NPE density estimator without appending/training data."""
    theta_dim = int(checkpoint["theta_dim"])
    x_dim = int(checkpoint["x_dim"])
    density_name = str(checkpoint["density_estimator_name"])
    builder = posterior_nn(density_name)
    density_estimator = builder(
        torch.zeros((2, theta_dim), dtype=torch.float32),
        torch.zeros((2, x_dim), dtype=torch.float32),
    )
    density_estimator.load_state_dict(checkpoint["density_estimator_state_dict"])
    density_estimator.eval()
    inference = FixedSplitNPE_C(
        prior=make_scaled_theta_prior("glu", theta_dim),
        density_estimator=density_name,
        device="cpu",
    )
    return inference.build_posterior(density_estimator)


def reference_input(checkpoint: dict[str, Any], observations: dict[str, np.ndarray]) -> torch.Tensor:
    x_scaled = transform_x_obs_with_fitted_scaler(
        x_obs=observations["x_obs"],
        mask=observations["mask"],
        x_scaler=checkpoint["x_scaler"],
        transform=str(checkpoint["x_scaling_metadata"]["transform"]),
    )
    mask = torch.tensor(observations["mask"], dtype=torch.float32)
    if checkpoint["method"] == "npe_zero_imputation":
        return zero_impute(x_obs_scaled=x_scaled, mask=mask)
    if checkpoint["method"] == "npe_mask_augmentation":
        return make_zero_imputed_mask_augmented_input(x_obs_scaled=x_scaled, mask=mask)
    raise ValueError(f"Unsupported checkpoint method: {checkpoint['method']!r}.")


def evaluate_checkpoint(
    checkpoint_file: Path,
    reference_file: Path,
    output_file: Path,
    *,
    num_posterior_samples: int,
    max_sampling_time: float,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    observations, reference_samples, reference_metadata, _ = (
        load_glu_mcar_partial_data_reference_posteriors_hdf5(reference_file)
    )
    if checkpoint["problem"] != "glu" or reference_metadata["problem"] != "glu":
        raise ValueError("This evaluator only supports GLU artifacts and checkpoints.")
    if int(checkpoint["theta_dim"]) != reference_samples.shape[2]:
        raise ValueError("Checkpoint and reference theta dimensions differ.")
    if num_posterior_samples != reference_samples.shape[1]:
        raise ValueError(
            "num_posterior_samples must equal the reference sample count "
            f"({reference_samples.shape[1]}), got {num_posterior_samples}."
        )
    x_ref = reference_input(checkpoint, observations)
    if x_ref.shape[1] != int(checkpoint["x_dim"]):
        raise ValueError("Checkpoint input dimension does not match method preprocessing.")

    seed = int(checkpoint["seed"])
    set_all_seeds(seed + 30_000)
    posterior = rebuild_posterior(checkpoint)
    start = time.perf_counter()
    samples_scaled, failures, fallbacks, fallback_used = sample_posteriors_once(
        posterior=posterior,
        x_eval=x_ref,
        num_posterior_samples=num_posterior_samples,
        seed=seed + 30_000,
        reject_outside_prior=bool(checkpoint["config"].get("sampling", {}).get("reject_outside_prior", True)),
        max_sampling_time=max_sampling_time,
        return_num_sampling_failures=True,
    )
    sampling_time_sec = time.perf_counter() - start
    scaled_np = samples_scaled.detach().cpu().numpy()
    theta_dim = scaled_np.shape[-1]
    samples = checkpoint["theta_scaler"].inverse_transform(
        scaled_np.reshape(-1, theta_dim)
    ).reshape(scaled_np.shape)
    reference_metrics_cfg = checkpoint["config"].get("reference_metrics", {})
    rows = compute_reference_metrics(
        estimator_samples=samples,
        reference_samples=reference_samples,
        seed=seed + 40_000,
        max_samples_per_observation=reference_samples.shape[1],
        c2st_max_samples_per_observation=int(
            reference_metrics_cfg.get("c2st_max_samples_per_observation", 2_000)
        ),
        c2st_n_folds=int(reference_metrics_cfg.get("c2st_n_folds", 3)),
        c2st_hidden_layer_scale=int(
            reference_metrics_cfg.get("c2st_hidden_layer_scale", 5)
        ),
        c2st_max_iter=int(reference_metrics_cfg.get("c2st_max_iter", 10_000)),
        progress=True,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_file.with_suffix(".h5"), "w") as h5:
        h5.create_dataset("theta_true", data=observations["theta_true"])
        h5.create_dataset("x_obs", data=observations["x_obs"])
        h5.create_dataset("mask", data=observations["mask"])
        h5.create_dataset("theta_posterior", data=samples)
        h5.attrs["reference_metric_target"] = "partial_data_x_obs_mask_reference"
        h5.attrs["reference_path"] = str(reference_file)
        h5.attrs["checkpoint_path"] = str(checkpoint_file)

    result = {
        "reference_metric_target": "partial_data_x_obs_mask_reference",
        "checkpoint_path": str(checkpoint_file),
        "reference_path": str(reference_file),
        "method": checkpoint["method"],
        "seed": seed,
        "num_reference_eval": int(samples.shape[0]),
        "num_reference_posterior_samples": int(samples.shape[1]),
        "reference_mask_missing_fraction": float(1.0 - observations["mask"].mean()),
        "sampling_time_sec": sampling_time_sec,
        "num_sampling_failures": int(failures),
        "num_sampling_fallbacks": int(fallbacks),
        "fallback_sampling_used": bool(fallback_used),
        "reference_metrics": rows,
        "reference_c2st_accuracy": [float(row["c2st_accuracy"]) for row in rows],
        "reference_posterior_mean_shift": [float(row["posterior_mean_shift"]) for row in rows],
        "reference_covariance_trace_ratio": [float(row["covariance_trace_ratio"]) for row in rows],
    }
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def main() -> None:
    args = parse_args()
    methods = args.method or list(METHODS)
    budgets = args.budget or list(BUDGETS)
    fractions = args.fraction or list(FRACTIONS)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for fraction in fractions:
        ref_path = reference_path(args.reference_root, fraction)
        if not ref_path.exists():
            raise FileNotFoundError(f"Missing partial-data reference: {ref_path}")
        for method in methods:
            for budget in budgets:
                for seed in SEEDS:
                    checkpoint = checkpoint_path(
                        args.outputs_root, METHODS[method], budget, fraction, seed
                    )
                    output = checkpoint.parent / "partial_data_reference_evaluation.json"
                    if not checkpoint.exists():
                        failures.append({"checkpoint_path": str(checkpoint), "error": "missing"})
                        continue
                    try:
                        result = evaluate_checkpoint(
                            checkpoint, ref_path, output,
                            num_posterior_samples=args.num_posterior_samples,
                            max_sampling_time=args.max_sampling_time,
                        )
                        result.update({"budget": budget, "fraction": fraction})
                        results.append(result)
                    except Exception as error:  # retain the campaign manifest on failures
                        failures.append({"checkpoint_path": str(checkpoint), "error": repr(error)})
    manifest = {"results": results, "failures": failures}
    manifest_path = args.outputs_root / (
        "partial_data_reference_evaluation_glu_mcar_"
        f"{'-'.join(methods)}_{'-'.join(budgets)}.json"
    )
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Evaluated {len(results)} checkpoints; failures={len(failures)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
