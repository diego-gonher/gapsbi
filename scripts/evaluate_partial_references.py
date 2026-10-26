"""Recompute full- and partial-reference metrics from saved estimator samples.

Current supported scope: GLU, GLM, OUP, and Lotka--Volterra under their
validated partial-reference configurations,
using saved posterior draws.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from gapsbi.evaluation.reference_metrics import (
    FINAL_REFERENCE_AGGREGATION_POLICY,
    aggregate_final_reference_metrics,
    compute_reference_metrics,
)
from gapsbi.references import (
    load_partial_data_reference_posteriors_hdf5,
    load_reference_posteriors_hdf5,
)

METHODS = {
    "zero": "npe_zero_imputation",
    "zero_mask": "npe_mask_augmentation",
    "learned_constant": "npe_learned_constant_imputation",
    "probabilistic": "npe_probabilistic_learned_imputation",
    "masked_transformer": "npe_masked_transformer_embedding",
}
BUDGETS = ("low", "mid", "high")
FRACTIONS = (0.10, 0.25, 0.50)
DEFAULT_SEEDS = (101, 202, 303, 404, 505)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problem", required=True)
    p.add_argument("--mechanism", required=True)
    p.add_argument("--fraction", type=float, action="append")
    p.add_argument("--method", choices=tuple(METHODS), action="append")
    p.add_argument("--budget", choices=BUDGETS, action="append")
    p.add_argument("--seeds", nargs="+", default=["all"])
    p.add_argument("--outputs-root", type=Path, default=Path("."))
    p.add_argument("--partial-reference-root", type=Path,
                   default=Path("references/partial_data_references_v1"))
    p.add_argument("--full-reference-path", type=Path)
    p.add_argument("--overwrite", action="store_true",
                   help="Replace an existing *_with_partial_references result.")
    p.add_argument("--refresh-aggregates", action="store_true",
                   help="Refresh saved aggregates from existing per-observation metric rows only.")
    p.add_argument("--partial-only", action="store_true",
                   help="Compute only new partial-reference metrics; never recompute full-reference metrics.")
    p.add_argument("--manifest-path", type=Path,
                   help="Optional path for this invocation's campaign manifest.")
    return p.parse_args()


def _digest(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array)
    return hashlib.sha256(a.view(np.uint8)).hexdigest()


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _experiment_stem(problem: str, mechanism: str, fraction: float) -> str:
    eps = f"eps{int(round(fraction * 100)):03d}"
    if problem == "lotka_volterra" and mechanism == "mcar":
        return f"lotka_volterra_time_block_mcar_{eps}"
    if problem == "lotka_volterra" and mechanism == "mnar":
        return f"lotka_volterra_log_total_mnar_{eps}"
    return f"glm_raw_{mechanism}_{eps}" if problem == "glm" else f"{problem}_{mechanism}_{eps}"


def _paths(root: Path, method: str, budget: str, mechanism: str, fraction: float, seed: int, *, problem: str = "glu", partial_only: bool = False) -> tuple[Path, Path]:
    d = root / f"outputs_{budget}_sim_budget" / METHODS[method] / f"{problem}/{mechanism}" / _experiment_stem(problem, mechanism, fraction) / f"seed_{seed}"
    name = "reference_metrics_partial_reference_only.json" if partial_only else "reference_metrics_with_partial_references.json"
    return d / "reference_posterior_samples.h5", d / name


def _partial_path(root: Path, mechanism: str, fraction: float, *, problem: str = "glu") -> Path:
    if (problem, mechanism, fraction) == ("lotka_volterra", "mnar", .50):
        return root / "lotka_volterra_mnar_eps050_successful_references.h5"
    return root / f"{problem}_{mechanism}_eps{int(round(fraction * 100)):03d}_references.h5"


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compatibility wrapper for the centralized final-benchmark policy."""
    return aggregate_final_reference_metrics(rows)


def _load_completed_output(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        result = json.load(f)
    if result.get("schema") != "gapsbi_reference_comparison_v1":
        raise ValueError(f"{path}: unrecognized comparison-result schema.")
    for name in ("full_reference", "partial_reference"):
        rows = result.get(name, {}).get("rows")
        if not isinstance(rows, list) or len(rows) != 10:
            raise ValueError(f"{path}: incomplete {name} metric rows.")
    return result


def refresh_aggregates(path: Path) -> dict[str, Any]:
    result = _load_completed_output(path)
    for target in ("full_reference", "partial_reference"):
        result[target]["aggregate"] = _aggregate(result[target]["rows"])
    result["aggregate_policy"] = FINAL_REFERENCE_AGGREGATION_POLICY
    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def load_compatible_samples(
    sample_path: Path, full_path: Path, partial_path: Path, *, method: str, fraction: float, seed: int,
    problem: str = "glu", partial_only: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Strictly establish that saved draws share the partial reference conditioning."""
    full_obs, full_ref, full_meta, _ = load_reference_posteriors_hdf5(full_path)
    part_obs, part_ref, part_meta, _ = load_partial_data_reference_posteriors_hdf5(partial_path)
    checkpoint_path = sample_path.with_name("model_checkpoint.pt")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"{sample_path}: sibling model checkpoint is missing.")
    with h5py.File(sample_path, "r") as h5:
        attrs = dict(h5.attrs)
        required = ("theta_true", "x_full", "x_obs", "mask", "theta_posterior", "theta_posterior_scaled")
        missing = [key for key in required if key not in h5]
        if missing:
            raise ValueError(f"{sample_path}: missing required datasets {missing}.")
        stored = {key: h5[key][...] for key in required}
    if attrs.get("problem") != problem or attrs.get("method") != METHODS[method] or int(attrs.get("seed", -1)) != seed:
        raise ValueError(f"{sample_path}: checkpoint/method provenance does not match selection.")
    if f"eps{int(round(fraction * 100)):03d}" not in str(attrs.get("dataset_path", "")):
        raise ValueError(f"{sample_path}: dataset path does not match selected fraction.")
    if attrs.get("reference_path") != str(full_path):
        raise ValueError(f"{sample_path}: not generated against the selected full reference.")
    for key, expected in (("theta_true", full_obs["theta_true"]), ("x_full", full_obs["x_full"])):
        if not np.array_equal(stored[key], expected):
            raise ValueError(f"{sample_path}: {key} is incompatible with reference observations/mask.")
    reference_indices = np.asarray(part_obs.get("reference_indices", np.arange(part_ref.shape[0])), dtype=int)
    if reference_indices.ndim != 1 or np.any(reference_indices < 0) or np.any(reference_indices >= stored["theta_posterior"].shape[0]):
        raise ValueError(f"{partial_path}: invalid original reference indices.")
    for key in ("theta_true", "x_full", "x_obs", "mask"):
        if not np.array_equal(stored[key][reference_indices], part_obs[key]):
            raise ValueError(f"{sample_path}: {key} is incompatible with sparse partial-reference mapping.")
    samples = stored["theta_posterior"][reference_indices] if partial_only else stored["theta_posterior"]
    if (not partial_only and (samples.shape != full_ref.shape or samples.shape != part_ref.shape)) or (
        partial_only and samples.shape != part_ref.shape
    ):
        raise ValueError(f"{sample_path}: posterior sample count/dimension differs from references.")
    expected_theta_transform = "standard" if problem in {"glm", "lotka_volterra"} else "minmax_minus_one_one"
    if attrs.get("theta_transform") != expected_theta_transform or attrs.get("x_transform") != "standard":
        raise ValueError(f"{sample_path}: unexpected {problem.upper()} parameter/input transforms.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for key, expected in (("problem", problem), ("method", METHODS[method]), ("seed", seed)):
        if checkpoint.get(key) != expected:
            raise ValueError(f"{checkpoint_path}: {key} does not match the selected evaluation.")
    if checkpoint.get("dataset_path") != attrs.get("dataset_path"):
        raise ValueError(f"{checkpoint_path}: dataset provenance differs from saved posterior samples.")
    if checkpoint.get("theta_dim") != samples.shape[-1]:
        raise ValueError(f"{checkpoint_path}: parameter dimension differs from saved posterior samples.")
    scaler = checkpoint.get("theta_scaler")
    if scaler is None:
        raise ValueError(f"{checkpoint_path}: theta scaler is unavailable; parameter units cannot be verified.")
    scaled = stored["theta_posterior_scaled"][reference_indices] if partial_only else stored["theta_posterior_scaled"]
    restored = scaler.inverse_transform(scaled.reshape(-1, samples.shape[-1])).reshape(samples.shape)
    if not np.array_equal(np.isfinite(restored), np.isfinite(samples)):
        raise ValueError(f"{sample_path}: scaled and unscaled posterior samples have different non-finite patterns.")
    if not np.allclose(restored, samples, rtol=1e-6, atol=1e-7, equal_nan=True):
        raise ValueError(f"{sample_path}: scaled and unscaled posterior samples disagree with checkpoint theta scaler.")
    return samples, {
        "estimator_samples_path": str(sample_path), "full_reference_path": str(full_path),
        "partial_reference_path": str(partial_path), "theta_true_sha256": _digest(stored["theta_true"]),
        "x_obs_sha256": _digest(stored["x_obs"]), "mask_sha256": _digest(stored["mask"]),
        "estimator_samples_sha256": _digest(samples), "original_reference_indices": reference_indices.tolist(), "reference_mask_seed": int(attrs["reference_mask_seed"]),
        "reference_mask_missing_fraction": float(attrs["reference_mask_missing_fraction"]),
        "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": _file_digest(checkpoint_path),
        "checkpoint_schema_version": int(checkpoint["schema_version"]),
        "parameter_space": "theta_unscaled",
    }


def evaluate_one(sample_path: Path, output_path: Path, full_path: Path, partial_path: Path, *, problem: str, method: str, mechanism: str, fraction: float, seed: int, overwrite: bool = False) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; use --overwrite to replace this new comparison output.")
    samples, provenance = load_compatible_samples(sample_path, full_path, partial_path, method=method, fraction=fraction, seed=seed, problem=problem)
    _, full_ref, _, _ = load_reference_posteriors_hdf5(full_path)
    _, partial_ref, _, _ = load_partial_data_reference_posteriors_hdf5(partial_path)
    kwargs = dict(seed=seed + 40_000, max_samples_per_observation=samples.shape[1],
                  c2st_max_samples_per_observation=2_000, c2st_n_folds=3,
                  c2st_hidden_layer_scale=5, c2st_max_iter=10_000, progress=True)
    full_rows = compute_reference_metrics(samples, full_ref, **kwargs)
    partial_rows = compute_reference_metrics(samples, partial_ref, **kwargs)
    full_invalid = [int(row["reference_index"]) for row in full_rows if not row["reference_metrics_valid"]]
    partial_invalid = [int(row["reference_index"]) for row in partial_rows if not row["reference_metrics_valid"]]
    if full_invalid != partial_invalid:
        raise ValueError("Full and partial reference metrics disagree on invalid estimator-sample observations.")
    result = {"schema": "gapsbi_reference_comparison_v1", "problem": problem, "mechanism": mechanism,
              "fraction": fraction, "method": METHODS[method], "budget": output_path.parents[5].name.replace("outputs_", "").replace("_sim_budget", ""),
              "seed": seed, "metric_settings": kwargs | {"progress": False}, "provenance": provenance,
              "aggregate_policy": FINAL_REFERENCE_AGGREGATION_POLICY,
              "sampling_failure_reference_indices": full_invalid,
              "full_reference": {"rows": full_rows, "aggregate": _aggregate(full_rows)},
              "partial_reference": {"rows": partial_rows, "aggregate": _aggregate(partial_rows)}}
    with output_path.open("w", encoding="utf-8") as f: json.dump(result, f, indent=2)
    return result


def evaluate_partial_only(sample_path: Path, output_path: Path, full_path: Path, partial_path: Path, *, problem: str, method: str, mechanism: str, fraction: float, seed: int, overwrite: bool = False) -> dict[str, Any]:
    """Compute only p(theta|x_obs,m) metrics; full results remain immutable."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; use --overwrite to replace this new partial-only output.")
    samples, provenance = load_compatible_samples(sample_path, full_path, partial_path, method=method, fraction=fraction, seed=seed, problem=problem, partial_only=True)
    partial_obs, partial_ref, partial_meta, _ = load_partial_data_reference_posteriors_hdf5(partial_path)
    original_indices = np.asarray(partial_obs.get("reference_indices", np.arange(partial_ref.shape[0])), dtype=int)
    kwargs = dict(seed=seed + 40_000, max_samples_per_observation=samples.shape[1],
                  c2st_max_samples_per_observation=2_000, c2st_n_folds=3,
                  c2st_hidden_layer_scale=5, c2st_max_iter=10_000, progress=True)
    rows = compute_reference_metrics(samples, partial_ref, **kwargs)
    for row, original_index in zip(rows, original_indices, strict=True):
        row["reference_index"] = int(original_index)
    invalid = [int(row["reference_index"]) for row in rows if not row["reference_metrics_valid"]]
    result = {"schema": "gapsbi_partial_reference_only_v1", "problem": problem, "mechanism": mechanism,
              "fraction": fraction, "method": METHODS[method], "budget": output_path.parents[5].name.replace("outputs_", "").replace("_sim_budget", ""),
              "seed": seed, "metric_settings": kwargs | {"progress": False}, "provenance": provenance,
              "aggregate_policy": FINAL_REFERENCE_AGGREGATION_POLICY, "partial_reference": {
                  "rows": rows, "aggregate": _aggregate(rows), "original_reference_indices": original_indices.tolist(),
                  "missing_original_reference_indices": sorted(set(range(10)) - set(original_indices.tolist())),
                  "reference_artifact_metadata": {key: partial_meta.get(key) for key in ("failed_reference_indices", "sparse_reference_artifact")},
              }, "sampling_failure_reference_indices": invalid}
    with output_path.open("w", encoding="utf-8") as f: json.dump(result, f, indent=2)
    return result


def main() -> None:
    a = parse_args()
    if (a.problem, a.mechanism) not in {
        ("glu", "mcar"), ("glu", "mnar"), ("glm", "mcar"), ("glm", "mnar"),
        ("oup", "mcar"), ("oup", "mnar"), ("lotka_volterra", "mcar"),
        ("lotka_volterra", "mnar"),
    }:
        raise NotImplementedError(
            "Supported configurations: GLU, GLM, and OUP under MCAR/MNAR, plus Lotka--Volterra/MCAR."
        )
    full_reference_path = a.full_reference_path or Path(
        f"references/reference_posteriors_v1/{a.problem}_references.h5"
    )
    seeds = DEFAULT_SEEDS if a.seeds == ["all"] else tuple(map(int, a.seeds))
    results, failures = [], []
    for fraction in a.fraction or FRACTIONS:
        partial = _partial_path(a.partial_reference_root, a.mechanism, fraction, problem=a.problem)
        for method in a.method or list(METHODS):
            for budget in a.budget or BUDGETS:
                for seed in seeds:
                    sample, output = _paths(a.outputs_root, method, budget, a.mechanism, fraction, seed, problem=a.problem, partial_only=a.partial_only)
                    try:
                        # Validate artifacts even when reusing an already-complete new result.
                        if output.exists() and a.refresh_aggregates and not a.overwrite:
                            load_compatible_samples(sample, full_reference_path, partial, method=method, fraction=fraction, seed=seed, problem=a.problem)
                            results.append(refresh_aggregates(output))
                        elif output.exists() and not a.overwrite:
                            load_compatible_samples(sample, full_reference_path, partial, method=method, fraction=fraction, seed=seed, problem=a.problem, partial_only=a.partial_only)
                            results.append(json.loads(output.read_text()) if a.partial_only else _load_completed_output(output))
                        else:
                            function = evaluate_partial_only if a.partial_only else evaluate_one
                            results.append(function(sample, output, full_reference_path, partial, problem=a.problem, method=method, mechanism=a.mechanism, fraction=fraction, seed=seed, overwrite=a.overwrite))
                    except Exception as e: failures.append({"samples": str(sample), "error": repr(e)})
    manifest = {"results": results, "failures": failures}
    path = a.manifest_path or (
        a.outputs_root / f"partial_reference_comparison_{a.problem}_{a.mechanism}_with_partial_references.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f: json.dump(manifest, f, indent=2)
    print(f"successful={len(results)} failures={len(failures)} manifest={path}")

if __name__ == "__main__": main()
