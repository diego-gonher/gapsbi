#!/usr/bin/env python3
"""Run one artifact-only standard L-C2ST feasibility pilot for GapSBI.

The pilot deliberately does not load estimator checkpoints or alter any metric
pipeline. It uses canonical held-out simulations as calibration pairs and the
saved posterior draws from a single already-trained estimator.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import sbi
import sklearn
import torch
from sklearn.exceptions import ConvergenceWarning

from gapsbi.evaluation.lc2st import (
    farthest_reference_indices,
    mask_augmented_context,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/canonical_v1/glu/mcar/glu_mcar_eps025_seed123.h5"
RUN = ROOT / "outputs_low_sim_budget/npe_mask_augmentation/glu/mcar/glu_mcar_eps025/seed_101"
PARTIAL_REFERENCE = ROOT / "references/partial_data_references_v1/glu_mcar_eps025_references.h5"


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")
    temporary.replace(path)


def h5_string(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def select_and_read_one_finite_draw_per_observation(dataset: h5py.Dataset, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stream HDF5 rows, retaining only one selected draw per observation."""
    rng = np.random.default_rng(seed)
    selected = []
    indices = []
    for index in range(dataset.shape[0]):
        row = dataset[index]
        valid = np.isfinite(row).all(axis=1)
        choices = np.flatnonzero(valid)
        if len(choices) == 0:
            raise ValueError(f"No finite posterior draw for calibration observation {index}.")
        draw = int(rng.choice(choices))
        selected.append(row[draw])
        indices.append(draw)
    return np.asarray(selected, dtype=np.float32), np.asarray(indices, dtype=np.int64)


def reference_metrics() -> dict[str, list[dict]]:
    path = RUN / "reference_metrics_with_partial_references.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {target: data[target]["rows"] for target in ("full_reference", "partial_reference")}


def validate_and_load() -> dict:
    required = ("posterior_samples.h5", "reference_posterior_samples.h5", "reference_metrics_with_partial_references.json")
    missing = [str(RUN / name) for name in required if not (RUN / name).exists()]
    if missing:
        raise FileNotFoundError("Required saved estimator artifact(s) unavailable: " + ", ".join(missing))
    if not PARTIAL_REFERENCE.exists():
        raise FileNotFoundError(f"Partial reference artifact unavailable: {PARTIAL_REFERENCE}")

    with h5py.File(DATASET, "r") as canonical:
        theta = canonical["test/theta"][:].astype(np.float32)
        x_obs = canonical["test/x_obs"][:].astype(np.float32)
        mask = canonical["test/mask"][:].astype(np.float32)
    context = mask_augmented_context(x_obs, mask).astype(np.float32)
    if theta.shape[0] != 1000:
        raise ValueError(f"Expected 1,000 canonical test observations, found {theta.shape[0]}.")

    with h5py.File(RUN / "posterior_samples.h5", "r") as stored:
        samples = stored["theta_posterior"]
        theta_true = stored["theta_true"][:]
        dataset_path = h5_string(stored.attrs["dataset_path"])
        if samples.shape != (1000, 1000, theta.shape[1]):
            raise ValueError(f"Unexpected calibration posterior shape {samples.shape}.")
        if not np.allclose(theta_true, theta, rtol=1e-5, atol=1e-6):
            raise ValueError("Saved calibration theta_true does not align with canonical test theta.")
        if not dataset_path.endswith("glu_mcar_eps025_seed123.h5"):
            raise ValueError(f"Unexpected calibration dataset provenance: {dataset_path}")

    with h5py.File(RUN / "reference_posterior_samples.h5", "r") as estimator_refs, h5py.File(PARTIAL_REFERENCE, "r") as partial:
        q_ref = estimator_refs["theta_posterior"][:].astype(np.float32)
        q_ref_scaled = estimator_refs["theta_posterior_scaled"][:]
        expected = ("theta_true", "x_full", "x_obs", "mask")
        for key in expected:
            if key not in estimator_refs or key not in partial["observations"]:
                raise KeyError(f"Missing {key} in reference artifact.")
            if not np.array_equal(estimator_refs[key][:], partial["observations"][key][:]):
                raise ValueError(f"Estimator and partial-reference {key} arrays are not exactly aligned.")
        if q_ref.shape != (10, 10_000, theta.shape[1]):
            raise ValueError(f"Unexpected reference posterior shape {q_ref.shape}.")
        if not np.all(np.isfinite(q_ref)) or not np.all(np.isfinite(q_ref_scaled)):
            raise ValueError("Reference estimator posterior draws contain non-finite values.")
        ref_context = mask_augmented_context(estimator_refs["x_obs"][:], estimator_refs["mask"][:]).astype(np.float32)

    return {
        "theta": theta,
        "context": context,
        "q_ref": q_ref,
        "ref_context": ref_context,
        "dataset_path": dataset_path,
        "validation": {
            "calibration_posterior_shape": [1000, 1000, int(theta.shape[1])],
            "reference_posterior_shape": [10, 10_000, int(theta.shape[1])],
            "reference_observation_mask_alignment_exact": True,
            "parameter_space": "raw_theta",
            "reference_indices": list(range(10)),
        },
    }


def run_one_seed(theta, context, q_calibration, q_reference, ref_context, classifier_seed, null_trials, control_permutation):
    # Import only after checking the artifact shapes, so unavailable saved draws fail fast.
    from sbi.diagnostics.lc2st import LC2ST

    common = dict(
        thetas=torch.as_tensor(theta),
        xs=torch.as_tensor(context),
        posterior_samples=torch.as_tensor(q_calibration),
        seed=classifier_seed,
        num_folds=1,
        num_ensemble=1,
        classifier="mlp",
        # sbi 0.26.1 double-normalizes x in its permutation-null path when True.
        # Inputs here are already bounded GLU values and masks, so keep this False.
        z_score=False,
        classifier_kwargs={"max_iter": 500, "n_iter_no_change": 50},
        num_trials_null=null_trials,
        permutation=True,
        device="cpu",
    )
    test = LC2ST(**common)
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        test.train_on_observed_data(seed=classifier_seed, verbosity=0)
        test.train_under_null_hypothesis(verbosity=0)
    fit_seconds = time.perf_counter() - started

    rows = []
    controls = []
    for index in range(q_reference.shape[0]):
        theta_o = torch.as_tensor(q_reference[index])
        x_o = torch.as_tensor(ref_context[index])
        statistic = test.get_statistic_on_observed_data(theta_o, x_o)
        null_stats = test.get_statistics_under_null_hypothesis(theta_o, x_o, verbosity=0)
        p_value = float((statistic < null_stats).mean())
        rows.append({"reference_index": index, "statistic": statistic, "p_value": p_value,
                     "null_statistic_mean": float(np.mean(null_stats)), "null_statistic_std": float(np.std(null_stats)),
                     "null_statistics": np.asarray(null_stats)})
        source = int(control_permutation[index])
        control_stat = test.get_statistic_on_observed_data(torch.as_tensor(q_reference[source]), x_o)
        control_null = test.get_statistics_under_null_hypothesis(torch.as_tensor(q_reference[source]), x_o, verbosity=0)
        controls.append({"reference_index": index, "mismatched_posterior_reference_index": source,
                         "statistic": control_stat, "p_value": float((control_stat < control_null).mean()),
                         "null_statistic_mean": float(np.mean(control_null)), "null_statistic_std": float(np.std(control_null))})
    return {"classifier_seed": classifier_seed, "fit_seconds": fit_seconds, "reference_results": rows,
            "positive_control_results": controls}


def run_synthetic_mismatch_control(theta, context, q_reference, ref_context, classifier_seed, null_trials, calibration_permutation):
    """Train a separate classifier against deliberately deranged calibration pairs."""
    from sbi.diagnostics.lc2st import LC2ST

    test = LC2ST(
        thetas=torch.as_tensor(theta), xs=torch.as_tensor(context),
        posterior_samples=torch.as_tensor(theta[calibration_permutation]),
        seed=classifier_seed, num_folds=1, num_ensemble=1, classifier="mlp", z_score=False,
        classifier_kwargs={"max_iter": 500, "n_iter_no_change": 50},
        num_trials_null=null_trials, permutation=True, device="cpu",
    )
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        test.train_on_observed_data(seed=classifier_seed, verbosity=0)
        test.train_under_null_hypothesis(verbosity=0)
    rows = []
    for index in range(q_reference.shape[0]):
        theta_o, x_o = torch.as_tensor(q_reference[index]), torch.as_tensor(ref_context[index])
        statistic = test.get_statistic_on_observed_data(theta_o, x_o)
        null_stats = test.get_statistics_under_null_hypothesis(theta_o, x_o, verbosity=0)
        rows.append({"reference_index": index, "statistic": statistic,
                     "p_value": float((statistic < null_stats).mean()),
                     "null_statistic_mean": float(np.mean(null_stats)),
                     "null_statistic_std": float(np.std(null_stats))})
    return {"fit_seconds": time.perf_counter() - started, "reference_results": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/diagnostics/lc2st_glu_mcar_eps025_zero_mask_low_seed101/results.json")
    parser.add_argument("--null-trials", type=int, default=20)
    parser.add_argument("--classifier-seeds", type=int, nargs=2, default=(101, 202))
    parser.add_argument("--draw-seed", type=int, default=7101)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} already exists; pass --overwrite to replace this pilot result.")
    if args.null_trials < 1:
        raise ValueError("--null-trials must be positive.")

    progress_path = args.output.with_name("progress.json")
    atomic_json(progress_path, {"status": "running", "started_unix": time.time(),
                                "completed_classifier_seeds": [], "output": str(args.output)})

    loaded = validate_and_load()
    # Generate only the reproducible HDF5 draw coordinates first, then read exactly those draws.
    with h5py.File(RUN / "posterior_samples.h5", "r") as stored:
        q_calibration, draw_indices = select_and_read_one_finite_draw_per_observation(
            stored["theta_posterior"], args.draw_seed
        )
    if not np.all(np.isfinite(q_calibration)):
        raise ValueError("Selected calibration posterior draws are non-finite.")

    # A visual local mismatch control. It is descriptive only and is superseded
    # by the calibration-pair derangement below as the actual sensitivity check.
    control_permutation = farthest_reference_indices(loaded["q_ref"])
    calibration_permutation = np.roll(np.arange(loaded["theta"].shape[0]), 1)
    started = time.perf_counter()
    seed_results = []
    for seed in args.classifier_seeds:
        run = run_one_seed(
            loaded["theta"], loaded["context"], q_calibration, loaded["q_ref"], loaded["ref_context"],
            seed, args.null_trials, control_permutation
        )
        run["synthetic_mismatch_control"] = run_synthetic_mismatch_control(
            loaded["theta"], loaded["context"], loaded["q_ref"], loaded["ref_context"],
            seed, args.null_trials, calibration_permutation
        )
        seed_results.append(run)
        atomic_json(progress_path, {"status": "running", "started_unix": time.time(),
                                    "completed_classifier_seeds": [r["classifier_seed"] for r in seed_results],
                                    "output": str(args.output)})
    result = {
        "schema": "gapsbi_lc2st_pilot_v1",
        "purpose": "standard_LC2ST_feasibility_pilot",
        "configuration": {"problem": "glu", "mechanism": "mcar", "missing_fraction": 0.25,
                          "method": "npe_mask_augmentation", "budget": "low", "npe_seed": 101},
        "implementation": {"sbi_version": sbi.__version__, "api": "sbi.diagnostics.lc2st.LC2ST",
                           "classifier": "sklearn.neural_network.MLPClassifier", "num_null_trials": args.null_trials,
                           "z_score": False, "device": "cpu"},
        "calibration": {"dataset_path": loaded["dataset_path"], "split": "test", "num_observations": 1000,
                        "observation_indices": np.arange(1000), "posterior_draw_indices": draw_indices,
                        "draw_selection_seed": args.draw_seed, "context": "concat(x_obs, mask)"},
        "artifact_validation": loaded["validation"],
        "reference_evaluation": {"num_observations": 10,
                                 "positive_control": "farthest_saved_reference_posterior_mean",
                                 "positive_control_permutation": control_permutation,
                                 "synthetic_mismatch_control": "calibration_theta_deranged_relative_to_context",
                                 "synthetic_mismatch_calibration_permutation": calibration_permutation,
                                 "existing_c2st": reference_metrics()},
        "classifier_runs": seed_results,
        "runtime_seconds": time.perf_counter() - started,
        "environment": {"python": sys.version, "numpy": np.__version__, "sklearn": sklearn.__version__,
                        "torch": torch.__version__, "platform": platform.platform()},
        "limitations": ["Twenty permutation-null trials give p-values on a 0.05 grid; these are preliminary.",
                        "Classifier-seed stability is assessed with two seeds only.",
                        "This standard L-C2ST pilot uses saved posterior draws; it does not test flow-based L-C2ST-NF."],
    }
    atomic_json(args.output, result)
    atomic_json(progress_path, {"status": "completed", "completed_classifier_seeds": list(args.classifier_seeds),
                                "output": str(args.output), "finished_unix": time.time()})
    print(args.output)


if __name__ == "__main__":
    main()
