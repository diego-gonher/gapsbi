"""Grid posterior evaluation for experimental full-data OUP NRE-A checkpoints."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import yaml
from gapsbi.evaluation.reference_metrics import compute_reference_metrics, aggregate_final_reference_metrics
from gapsbi.methods.sbi_nre import load_nre_a_full_data_checkpoint, evaluate_nre_a_oup_grid, sample_nre_a_oup_grid
from gapsbi.references import load_reference_posteriors_hdf5


def evaluate_checkpoint(checkpoint_path: Path, config: dict, repository_root: Path = Path(".")) -> dict:
    checkpoint, estimator = load_nre_a_full_data_checkpoint(checkpoint_path, repository_root=repository_root)
    settings = config["evaluation"]
    observations, reference, metadata, _ = load_reference_posteriors_hdf5(settings["reference_path"])
    if metadata.get("problem") != "oup" or reference.shape != (10, 10_000, 2):
        raise ValueError("Expected the canonical OUP full-data reference artifact.")
    x = checkpoint["x_scaler"].transform(observations["x_full"])
    resolutions = [int(value) for value in settings["grid_resolutions"]]
    selected = int(settings["selected_grid_resolution"])
    if selected not in resolutions:
        raise ValueError("selected_grid_resolution must be tested for convergence.")
    grids, diagnostics = [], []
    for index in range(len(x)):
        by_resolution = [evaluate_nre_a_oup_grid(estimator, theta_scaler=checkpoint["theta_scaler"], x_scaled=x[index], resolution=r, batch_size=int(settings["grid_batch_size"])) for r in resolutions]
        chosen = next(grid for grid in by_resolution if grid["resolution"] == selected)
        grids.append(chosen)
        diagnostics.append({"reference_index": index, "resolutions": resolutions,
            "means": [g["mean"].tolist() for g in by_resolution],
            "covariance_traces": [float(np.trace(g["cov"])) for g in by_resolution],
            "boundary_masses": [g["boundary_mass"] for g in by_resolution],
            "mean_l2_deltas": [float(np.linalg.norm(g["mean"] - chosen["mean"])) for g in by_resolution],
            "covariance_trace_deltas": [float(abs(np.trace(g["cov"]) - np.trace(chosen["cov"]))) for g in by_resolution]})
    samples = np.stack([sample_nre_a_oup_grid(grid, num_samples=reference.shape[1], seed=int(checkpoint["seed"]) + 40_000 + i) for i, grid in enumerate(grids)])
    if not np.isfinite(samples).all() or np.any(samples < [0., -2.]) or np.any(samples > [2., 3.]):
        raise FloatingPointError("Grid posterior samples are nonfinite or outside OUP support.")
    metric_cfg = settings["reference_metrics"]
    rows = compute_reference_metrics(samples, reference, seed=int(checkpoint["seed"]), max_samples_per_observation=10_000,
        c2st_max_samples_per_observation=int(metric_cfg["c2st_max_samples_per_observation"]), c2st_n_folds=int(metric_cfg["c2st_n_folds"]), c2st_hidden_layer_scale=int(metric_cfg["c2st_hidden_layer_scale"]), c2st_max_iter=int(metric_cfg["c2st_max_iter"]))
    result = {"method": "nre_a_full_data", "seed": int(checkpoint["seed"]), "reference_path": settings["reference_path"], "posterior_target": "theta_given_x_full", "grid_log_ratio_convention": "log_p_x_given_theta_minus_log_p_x", "grid_prior": "uniform_raw_oup_box", "selected_grid_resolution": selected, "grid_convergence": diagnostics, "reference_metrics": rows, "reference_metric_aggregate": aggregate_final_reference_metrics(rows)}
    output = checkpoint_path.parent
    with h5py.File(output / "reference_posterior_samples.h5", "w") as h5:
        h5.create_dataset("theta_true", data=observations["theta_true"])
        h5.create_dataset("x_full", data=observations["x_full"])
        h5.create_dataset("theta_posterior", data=samples)
        h5.attrs["method"] = "nre_a_full_data"; h5.attrs["seed"] = int(checkpoint["seed"]); h5.attrs["grid_resolution"] = selected
    (output / "reference_evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = evaluate_checkpoint(args.checkpoint, yaml.safe_load(args.config.read_text()), args.repository_root)
    print(json.dumps(result["reference_metric_aggregate"], sort_keys=True))
if __name__ == "__main__": main()
