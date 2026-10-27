"""Reusable, artifact-oriented standard L-C2ST evaluation.

This deliberately remains an experimental diagnostic.  It uses SBI's standard
``LC2ST`` implementation, never L-C2ST-NF, and accepts the raw GapSBI
``[x_obs, mask]`` context independently of an estimator's internal encoding.
"""
from __future__ import annotations

import hashlib
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sbi.diagnostics.lc2st import LC2ST

from gapsbi.evaluation.checkpoint_adapters import NPECheckpointAdapter, load_npe_checkpoint_adapter
from gapsbi.evaluation.lc2st import mask_augmented_context
from gapsbi.io import load_gapsbi_hdf5
from gapsbi.references import load_partial_data_reference_posteriors_hdf5


SCHEMA_NAME = "gapsbi_standard_lc2st"
SCHEMA_VERSION = 1


def corrected_permutation_p_value(statistic: float, null_statistics: np.ndarray) -> float:
    """Return the finite-sample corrected upper-tail permutation p-value."""
    null = np.asarray(null_statistics, dtype=float).reshape(-1)
    if not np.isfinite(statistic) or not np.all(np.isfinite(null)):
        raise ValueError("Observed and null L-C2ST statistics must be finite.")
    return float((1 + np.count_nonzero(null >= statistic)) / (null.size + 1))


def split_paired_calibration_indices(size: int, holdout_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split observation IDs, never individual examples from a paired row."""
    if size < 2 or not 0.0 < holdout_fraction < 1.0:
        raise ValueError("Need at least two rows and a holdout fraction in (0, 1).")
    n_holdout = min(max(1, round(size * holdout_fraction)), size - 1)
    order = np.random.default_rng(seed).permutation(size)
    return np.sort(order[n_holdout:]), np.sort(order[:n_holdout])


def _standardizer_fit(theta: np.ndarray, context: np.ndarray) -> dict[str, list[float]]:
    # Both classifier classes share a conditioning context for each
    # calibration row, so fitting its marginal once avoids duplicating rows.
    mean = np.concatenate((theta.mean(axis=0), context.mean(axis=0)))
    scale = np.concatenate((theta.std(axis=0), context.std(axis=0)))
    scale[scale == 0] = 1.0
    return {"mean": mean.tolist(), "scale": scale.tolist()}


def _z(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (values - mean) / scale


def _row_fingerprints(theta: np.ndarray, x_full: np.ndarray, mask: np.ndarray) -> set[str]:
    return {
        hashlib.sha256(np.ascontiguousarray(np.concatenate((t, x, m))).view(np.uint8)).hexdigest()
        for t, x, m in zip(theta, x_full, mask, strict=True)
    }


def assert_calibration_separate_from_evaluation(calibration: dict[str, np.ndarray], observations: dict[str, np.ndarray]) -> None:
    """Reject exact prior-predictive rows reused as evaluation observations."""
    overlap = _row_fingerprints(calibration["theta"], calibration["x_full"], calibration["mask"]) & _row_fingerprints(
        observations["theta_true"], observations["x_full"], observations["mask"]
    )
    if overlap:
        raise ValueError("Calibration and evaluation observations overlap.")


def build_calibration_classes(adapter: NPECheckpointAdapter, calibration: dict[str, np.ndarray], posterior_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return aligned q samples, prior-joint theta values, and raw contexts."""
    context = mask_augmented_context(calibration["x_obs"], calibration["mask"]).astype(np.float32)
    posterior = adapter.sample(calibration["x_obs"], calibration["mask"], num_posterior_samples=1, seed=posterior_seed)[:, 0]
    prior = np.asarray(calibration["theta"], dtype=np.float32)
    if posterior.shape != prior.shape or not np.all(np.isfinite(posterior)):
        raise ValueError("Calibration posterior samples are invalid or misaligned.")
    return posterior.astype(np.float32), prior, context


def _null_array(value: Any) -> np.ndarray:
    # SBI 0.26.1 returns ndarray; tolerate a future score wrapper without
    # altering the persisted result contract.
    return np.asarray(getattr(value, "scores", value), dtype=float).reshape(-1)


def run_standard_lc2st(*, calibration_q: np.ndarray, calibration_prior: np.ndarray, calibration_context: np.ndarray, evaluation_q: np.ndarray, evaluation_context: np.ndarray, reference_indices: np.ndarray, classifier_seed: int, num_permutations: int = 199, alpha: float = 0.05, classifier_max_iter: int = 500, classifier_n_iter_no_change: int = 50) -> dict[str, Any]:
    """Fit one standard L-C2ST and evaluate local MSE at reference contexts."""
    q, prior, context = map(np.asarray, (calibration_q, calibration_prior, calibration_context))
    evaluation_q, evaluation_context, reference_indices = map(np.asarray, (evaluation_q, evaluation_context, reference_indices))
    if q.ndim != 2 or prior.shape != q.shape or context.ndim != 2 or context.shape[0] != q.shape[0]:
        raise ValueError("Calibration q, prior, and context must be aligned two-dimensional arrays.")
    if evaluation_q.ndim != 3 or evaluation_q.shape[0] != evaluation_context.shape[0] or evaluation_q.shape[2] != q.shape[1]:
        raise ValueError("Evaluation samples and contexts have incompatible shapes.")
    if reference_indices.shape != (evaluation_q.shape[0],) or len(np.unique(reference_indices)) != reference_indices.size:
        raise ValueError("reference_indices must be unique and match evaluation rows.")
    if num_permutations < 1 or not np.all(np.isfinite(q)) or not np.all(np.isfinite(prior)) or not np.all(np.isfinite(evaluation_q)):
        raise ValueError("L-C2ST requires finite samples and at least one permutation.")
    scaler = _standardizer_fit(np.concatenate((q, prior)), context)
    mean, scale, d = np.asarray(scaler["mean"]), np.asarray(scaler["scale"]), q.shape[1]
    qz, priorz, contextz = _z(q, mean[:d], scale[:d]), _z(prior, mean[:d], scale[:d]), _z(context, mean[d:], scale[d:])
    eval_qz, eval_contextz = _z(evaluation_q, mean[:d], scale[:d]), _z(evaluation_context, mean[d:], scale[d:])
    test = LC2ST(thetas=torch.as_tensor(priorz, dtype=torch.float32), xs=torch.as_tensor(contextz, dtype=torch.float32), posterior_samples=torch.as_tensor(qz, dtype=torch.float32), seed=int(classifier_seed), num_folds=1, num_ensemble=1, z_score=False, classifier_kwargs={"max_iter": classifier_max_iter, "n_iter_no_change": classifier_n_iter_no_change}, num_trials_null=int(num_permutations), permutation=True, device="cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        fit_started = time.perf_counter()
        test.train_on_observed_data(seed=int(classifier_seed), verbosity=0)
        observed_fit_seconds = time.perf_counter() - fit_started
        permutation_started = time.perf_counter()
        test.train_under_null_hypothesis(verbosity=0)
        permutation_fit_seconds = time.perf_counter() - permutation_started
    local_started = time.perf_counter()
    results = []
    for index, samples, x in zip(reference_indices, eval_qz, eval_contextz, strict=True):
        theta, x_tensor = torch.as_tensor(samples, dtype=torch.float32), torch.as_tensor(x[None], dtype=torch.float32)
        statistic = float(test.get_statistic_on_observed_data(theta, x_tensor))
        null = _null_array(test.get_statistics_under_null_hypothesis(theta, x_tensor, verbosity=0))
        p_value = corrected_permutation_p_value(statistic, null)
        results.append({"reference_index": int(index), "local_mse": statistic, "null_local_mse": null.tolist(), "p_value": p_value, "reject": bool(p_value <= alpha)})
    return {"schema_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION, "target": "q_theta_given_x_obs_mask", "classifier": {"implementation": "sbi.diagnostics.lc2st.LC2ST", "z_score": False, "seed": int(classifier_seed), "permutation_seed": int(classifier_seed), "num_permutations": int(num_permutations), "alpha": float(alpha), "standardizer": scaler}, "timing_seconds": {"observed_classifier_fit": observed_fit_seconds, "permutation_classifier_fit": permutation_fit_seconds, "local_evaluation": time.perf_counter() - local_started}, "reference_results": results}


def load_calibration_pairs(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    dataset, metadata = load_gapsbi_hdf5(path)
    if metadata.get("calibration_role") != "lc2st_prior_predictive_calibration" or metadata.get("mechanism") != "mcar":
        raise ValueError("Not an MCAR L-C2ST calibration dataset.")
    if dataset["train"]["theta"].shape[0] or dataset["val"]["theta"].shape[0]:
        raise ValueError("L-C2ST calibration data must use only the test split.")
    return dataset["test"], metadata


def run_from_artifacts(*, checkpoint_path: str | Path, calibration_path: str | Path, partial_reference_path: str | Path, repository_root: str | Path, classifier_seed: int, calibration_posterior_seed: int, reference_posterior_seed: int, num_reference_posterior_samples: int, num_permutations: int = 199, alpha: float = 0.05, classifier_max_iter: int = 500, classifier_n_iter_no_change: int = 50) -> dict[str, Any]:
    adapter = load_npe_checkpoint_adapter(checkpoint_path, repository_root=repository_root)
    calibration, calibration_meta = load_calibration_pairs(calibration_path)
    observations, _reference, reference_meta, _diagnostics = load_partial_data_reference_posteriors_hdf5(partial_reference_path)
    if calibration_meta.get("task") != adapter.problem or reference_meta.get("problem") != adapter.problem:
        raise ValueError("Checkpoint, calibration, and partial-reference task must agree.")
    assert_calibration_separate_from_evaluation(calibration, observations)
    posterior_started = time.perf_counter()
    q, prior, context = build_calibration_classes(adapter, calibration, calibration_posterior_seed)
    calibration_posterior_seconds = time.perf_counter() - posterior_started
    posterior_started = time.perf_counter()
    eval_q = adapter.sample(observations["x_obs"], observations["mask"], num_posterior_samples=num_reference_posterior_samples, seed=reference_posterior_seed)
    reference_posterior_seconds = time.perf_counter() - posterior_started
    result = run_standard_lc2st(calibration_q=q, calibration_prior=prior, calibration_context=context, evaluation_q=eval_q, evaluation_context=mask_augmented_context(observations["x_obs"], observations["mask"]).astype(np.float32), reference_indices=observations.get("reference_indices", np.arange(eval_q.shape[0])), classifier_seed=classifier_seed, num_permutations=num_permutations, alpha=alpha, classifier_max_iter=classifier_max_iter, classifier_n_iter_no_change=classifier_n_iter_no_change)
    result["provenance"] = {"checkpoint": str(checkpoint_path), "calibration": str(calibration_path), "partial_reference": str(partial_reference_path), "task": adapter.problem, "calibration_metadata": calibration_meta, "reference_metadata": reference_meta, "posterior_sampling_seeds": {"calibration": calibration_posterior_seed, "reference": reference_posterior_seed}}
    result["timing_seconds"]["calibration_posterior_sampling"] = calibration_posterior_seconds
    result["timing_seconds"]["reference_posterior_sampling"] = reference_posterior_seconds
    return result


def write_standard_lc2st_result(path: str | Path, result: dict[str, Any], *, overwrite: bool = False) -> None:
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
