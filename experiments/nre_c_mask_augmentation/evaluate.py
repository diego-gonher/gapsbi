"""Evaluate OUP NRE-C mask augmentation against partial-data references."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

from gapsbi.evaluation.reference_metrics import aggregate_final_reference_metrics, compute_reference_metrics
from gapsbi.evaluation.nre_c_campaign import validate_oup_partial_reference_metadata
from gapsbi.methods.sbi_nre import (
    evaluate_nre_a_oup_grid,
    load_nre_c_mask_augmentation_checkpoint,
    prepare_mask_augmented_nre_input,
    sample_nre_a_oup_grid,
)
from gapsbi.references import load_partial_data_reference_posteriors_hdf5


def select_reference_positions(observations, requested_ids):
    ids = np.asarray(observations.get("reference_indices", np.arange(len(observations["x_obs"]))), dtype=int)
    requested = ids if requested_ids is None else np.asarray(requested_ids, dtype=int)
    positions = []
    for reference_id in requested:
        matches = np.flatnonzero(ids == reference_id)
        if len(matches) != 1:
            raise ValueError(f"Requested reference ID {reference_id} is not uniquely available.")
        positions.append(int(matches[0]))
    positions = np.asarray(positions, dtype=int)
    return positions, ids[positions]


def evaluate_checkpoint(checkpoint_path, config, *, reference_ids=None, repository_root=Path(".")):
    started = time.perf_counter()
    checkpoint, estimator = load_nre_c_mask_augmentation_checkpoint(checkpoint_path, repository_root=repository_root)
    evaluation = config["evaluation"]
    observations, reference_samples, reference_metadata, _ = load_partial_data_reference_posteriors_hdf5(evaluation["partial_reference_path"])
    validate_oup_partial_reference_metadata(
        reference_metadata, mechanism=config.get("mechanism", "mcar"),
        missing_fraction=float(config.get("missing_fraction", 0.1)),
    )
    positions, canonical_ids = select_reference_positions(observations, evaluation.get("reference_indices") if reference_ids is None else reference_ids)
    conditions = prepare_mask_augmented_nre_input(
        x_obs=observations["x_obs"][positions], mask=observations["mask"][positions], x_scaler=checkpoint["x_scaler"],
        transform=checkpoint["x_scaling_metadata"]["transform"],
    ).numpy()
    resolutions, selected_resolution = [int(value) for value in evaluation["grid_resolutions"]], int(evaluation["selected_grid_resolution"])
    if selected_resolution not in resolutions:
        raise ValueError("selected_grid_resolution must occur in grid_resolutions.")
    grids, diagnostics = [], []
    for reference_id, condition in zip(canonical_ids, conditions, strict=True):
        candidates = [evaluate_nre_a_oup_grid(estimator, theta_scaler=checkpoint["theta_scaler"], x_scaled=condition, resolution=resolution, batch_size=int(evaluation["grid_batch_size"])) for resolution in resolutions]
        selected = next(grid for grid in candidates if grid["resolution"] == selected_resolution)
        grids.append(selected)
        diagnostics.append({"reference_index": int(reference_id), "resolutions": resolutions,
                            "means": [grid["mean"].tolist() for grid in candidates],
                            "covariance_traces": [float(np.trace(grid["cov"])) for grid in candidates],
                            "boundary_masses": [grid["boundary_mass"] for grid in candidates],
                            "mean_l2_deltas": [float(np.linalg.norm(grid["mean"] - selected["mean"])) for grid in candidates],
                            "covariance_trace_deltas": [float(abs(np.trace(grid["cov"]) - np.trace(selected["cov"]))) for grid in candidates]})
    samples = np.stack([sample_nre_a_oup_grid(grid, num_samples=reference_samples.shape[1], seed=int(checkpoint["seed"]) + 40_000 + int(reference_id)) for reference_id, grid in zip(canonical_ids, grids, strict=True)])
    metrics = evaluation["reference_metrics"]
    rows = compute_reference_metrics(samples, reference_samples[positions], seed=int(checkpoint["seed"]), max_samples_per_observation=10_000,
                                     c2st_max_samples_per_observation=int(metrics["c2st_max_samples_per_observation"]),
                                     c2st_n_folds=int(metrics["c2st_n_folds"]), c2st_hidden_layer_scale=int(metrics["c2st_hidden_layer_scale"]),
                                     c2st_max_iter=int(metrics["c2st_max_iter"]))
    for row, reference_id in zip(rows, canonical_ids, strict=True):
        row["reference_index"] = int(reference_id)
    result = {"method": "nre_c_mask_augmentation", "seed": int(checkpoint["seed"]), "posterior_target": "theta_given_x_obs_mask",
              "augmentation": "zero_imputed_scaled_x_plus_mask", "reference_artifact": str(evaluation["partial_reference_path"]),
              "reference_ids": canonical_ids.tolist(), "reference_metadata": reference_metadata,
              "grid_log_ratio_convention": "log_p_x_given_theta_minus_log_p_x", "grid_prior": "uniform_raw_oup_box",
              "grid_batch_size": int(evaluation["grid_batch_size"]), "selected_grid_resolution": selected_resolution,
              "inference_runtime_sec": time.perf_counter() - started,
              "grid_convergence": diagnostics, "reference_metrics": rows, "reference_metric_aggregate": aggregate_final_reference_metrics(rows)}
    output_dir = Path(checkpoint_path).parent
    with h5py.File(output_dir / "reference_posterior_samples.h5", "w") as h5:
        for key in ("theta_true", "x_full", "x_obs", "mask"):
            h5.create_dataset(key, data=observations[key][positions])
        h5.create_dataset("reference_indices", data=canonical_ids)
        h5.create_dataset("theta_posterior", data=samples)
        h5.attrs["method"] = "nre_c_mask_augmentation"
        h5.attrs["grid_resolution"] = selected_resolution
        h5.attrs["posterior_target"] = "theta_given_x_obs_mask"
    (output_dir / "reference_evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-indices", type=int, nargs="*", default=None)
    args = parser.parse_args()
    result = evaluate_checkpoint(args.checkpoint, yaml.safe_load(args.config.read_text()), reference_ids=args.reference_indices)
    print(json.dumps(result["reference_metric_aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
