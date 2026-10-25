"""Focused contract tests for the saved-sample partial-reference evaluator."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler

from gapsbi.references import (
    generate_glu_mcar_partial_data_reference_posteriors,
    generate_glu_reference_posteriors,
    save_glu_mcar_partial_data_reference_posteriors_hdf5,
    save_reference_posteriors_hdf5,
)


SPEC = importlib.util.spec_from_file_location(
    "evaluate_partial_references", Path(__file__).parents[1] / "scripts/evaluate_partial_references.py"
)
assert SPEC is not None and SPEC.loader is not None
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)

ANALYSIS_SPEC = importlib.util.spec_from_file_location(
    "analyze_partial_reference_comparison",
    Path(__file__).parents[1] / "scripts/analyze_partial_reference_comparison.py",
)
assert ANALYSIS_SPEC is not None and ANALYSIS_SPEC.loader is not None
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
ANALYZER = importlib.util.module_from_spec(ANALYSIS_SPEC)
ANALYSIS_SPEC.loader.exec_module(ANALYZER)


def _compatible_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    full_obs, full_meta, full_data = generate_glu_reference_posteriors(
        num_observations=2, num_reference_samples=20, observation_seed=1, posterior_seed=2, dim=3
    )
    full_path = tmp_path / "glu_references.h5"
    save_reference_posteriors_hdf5(
        full_path, observations=full_obs, theta_samples=full_data["theta_samples"], metadata=full_meta
    )
    part_obs, part_meta, part_data = generate_glu_mcar_partial_data_reference_posteriors(
        num_observations=2, num_reference_samples=20, observation_seed=1, posterior_seed=3,
        dim=3, missing_fraction=0.25, reference_mask_seed=987654,
    )
    partial_path = tmp_path / "glu_mcar_eps025_references.h5"
    save_glu_mcar_partial_data_reference_posteriors_hdf5(
        partial_path, observations=part_obs, theta_samples=part_data["theta_samples"], metadata=part_meta
    )
    sample_path = tmp_path / "reference_posterior_samples.h5"
    samples = full_data["theta_samples"]
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(np.vstack([[-1.0] * 3, [1.0] * 3]))
    with h5py.File(sample_path, "w") as h5:
        h5.attrs.update({"problem": "glu", "method": "npe_zero_imputation", "seed": 101,
                         "dataset_path": "data/glu_mcar_eps025_seed123.h5", "reference_path": str(full_path),
                         "theta_transform": "minmax_minus_one_one", "x_transform": "standard",
                         "reference_mask_seed": 987654, "reference_mask_missing_fraction": 0.25})
        for key, value in {**part_obs, "theta_posterior": samples,
                           "theta_posterior_scaled": scaler.transform(samples.reshape(-1, 3)).reshape(samples.shape)}.items():
            h5.create_dataset(key, data=value)
    torch.save({"schema_version": 2, "problem": "glu", "method": "npe_zero_imputation", "seed": 101,
                "dataset_path": "data/glu_mcar_eps025_seed123.h5", "theta_dim": 3, "theta_scaler": scaler},
               sample_path.with_name("model_checkpoint.pt"))
    return sample_path, full_path, partial_path


def test_saved_samples_require_full_observation_mask_and_checkpoint_compatibility(tmp_path: Path) -> None:
    sample_path, full_path, partial_path = _compatible_artifacts(tmp_path)
    samples, provenance = EVALUATOR.load_compatible_samples(
        sample_path, full_path, partial_path, method="zero", fraction=0.25, seed=101
    )
    assert samples.shape == (2, 20, 3)
    assert provenance["parameter_space"] == "theta_unscaled"
    assert len(provenance["checkpoint_sha256"]) == 64

    with h5py.File(sample_path, "r+") as h5:
        h5["mask"][0, 0] = 1 - h5["mask"][0, 0]
    with pytest.raises(ValueError, match="mask is incompatible"):
        EVALUATOR.load_compatible_samples(sample_path, full_path, partial_path, method="zero", fraction=0.25, seed=101)


def test_saved_samples_reject_unit_mismatch(tmp_path: Path) -> None:
    sample_path, full_path, partial_path = _compatible_artifacts(tmp_path)
    with h5py.File(sample_path, "r+") as h5:
        h5["theta_posterior_scaled"][...] += 0.5
    with pytest.raises(ValueError, match="scaled and unscaled"):
        EVALUATOR.load_compatible_samples(sample_path, full_path, partial_path, method="zero", fraction=0.25, seed=101)


def test_saved_samples_allow_matching_nan_failure_blocks(tmp_path: Path) -> None:
    sample_path, full_path, partial_path = _compatible_artifacts(tmp_path)
    with h5py.File(sample_path, "r+") as h5:
        h5["theta_posterior"][1, ...] = np.nan
        h5["theta_posterior_scaled"][1, ...] = np.nan
    samples, _ = EVALUATOR.load_compatible_samples(
        sample_path, full_path, partial_path, method="zero", fraction=0.25, seed=101
    )
    assert np.isnan(samples[1]).all()


def test_aggregate_preserves_invalid_observations() -> None:
    rows = [
        {"reference_index": 0, "reference_metrics_valid": True,
         "c2st_accuracy": 0.5, "posterior_mean_shift": 0.1,
         "covariance_trace_ratio": 1.0},
        {"reference_index": 1, "reference_metrics_valid": False,
         "c2st_accuracy": float("nan"), "posterior_mean_shift": float("nan"),
         "covariance_trace_ratio": float("nan")},
    ]
    aggregate = EVALUATOR._aggregate(rows)
    assert aggregate["c2st_accuracy_finite_count"] == 1
    assert aggregate["invalid_reference_indices"] == [1]
    assert not aggregate["c2st_accuracy_all_observations_valid"]
    assert aggregate["c2st_accuracy"] == 0.5


def test_aggregate_uses_original_metric_reducers() -> None:
    rows = [
        {"reference_index": i, "reference_metrics_valid": True,
         "c2st_accuracy": value, "posterior_mean_shift": value,
         "covariance_trace_ratio": value}
        for i, value in enumerate((0.5, 1.0, 100.0))
    ]
    aggregate = EVALUATOR._aggregate(rows)
    assert aggregate["c2st_accuracy"] == pytest.approx((0.5 + 1.0 + 100.0) / 3)
    assert aggregate["posterior_mean_shift"] == 1.0
    assert aggregate["covariance_trace_ratio"] == 1.0


def test_aggregate_is_nan_when_no_finite_observations_remain() -> None:
    rows = [{"reference_index": 0, "reference_metrics_valid": False,
             "c2st_accuracy": float("nan"), "posterior_mean_shift": float("nan"),
             "covariance_trace_ratio": float("nan")}]
    aggregate = EVALUATOR._aggregate(rows)
    for metric in ("c2st_accuracy", "posterior_mean_shift", "covariance_trace_ratio"):
        assert aggregate[f"{metric}_finite_count"] == 0
        assert np.isnan(aggregate[metric])


def test_seed_summary_filters_nonfinite_seeds_and_uses_sample_sd() -> None:
    frame = pd.DataFrame([
        {"metric": "c2st", "method": "npe_zero_imputation", "fraction": 0.5,
         "budget": "low", "target": "full", "seed": 101, "value": 1.0,
         "configuration_complete": True, "n_valid_reference_observations": 10},
        {"metric": "c2st", "method": "npe_zero_imputation", "fraction": 0.5,
         "budget": "low", "target": "full", "seed": 202, "value": 3.0,
         "configuration_complete": True, "n_valid_reference_observations": 10},
        {"metric": "c2st", "method": "npe_zero_imputation", "fraction": 0.5,
         "budget": "low", "target": "full", "seed": 303, "value": float("nan"),
         "configuration_complete": False, "n_valid_reference_observations": 9},
    ])
    summary = ANALYZER.summarize(frame).iloc[0]
    assert summary["mean_full"] == pytest.approx(2.0)
    assert summary["std_full"] == pytest.approx(np.sqrt(2.0))
    assert summary["n_seeds_full"] == 2
    assert 1.96 * summary["std_full"] / np.sqrt(summary["n_seeds_full"]) == pytest.approx(1.96)


def test_mnar_paths_use_distinct_artifacts_and_outputs(tmp_path: Path) -> None:
    samples, result = EVALUATOR._paths(tmp_path, "zero", "low", "mnar", 0.25, 101)
    assert samples == tmp_path / "outputs_low_sim_budget/npe_zero_imputation/glu/mnar/glu_mnar_eps025/seed_101/reference_posterior_samples.h5"
    assert result.name == "reference_metrics_with_partial_references.json"
    assert EVALUATOR._partial_path(tmp_path, "mnar", 0.25).name == "glu_mnar_eps025_references.h5"


def test_oup_paths_and_scaler_convention_match_saved_artifacts(tmp_path: Path) -> None:
    samples, result = EVALUATOR._paths(
        tmp_path, "zero", "high", "mcar", 0.50, 101, problem="oup"
    )
    assert samples == tmp_path / (
        "outputs_high_sim_budget/npe_zero_imputation/oup/mcar/oup_mcar_eps050/"
        "seed_101/reference_posterior_samples.h5"
    )
    assert result.name == "reference_metrics_with_partial_references.json"
    assert EVALUATOR._partial_path(tmp_path, "mcar", 0.50, problem="oup").name == (
        "oup_mcar_eps050_references.h5"
    )
