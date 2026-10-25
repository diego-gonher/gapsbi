"""Regression coverage for non-mutating reference-aware benchmark inputs."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SPEC = importlib.util.spec_from_file_location(
    "reference_aware_plot_input", Path(__file__).parents[1] / "scripts/reference_aware_plot_input.py"
)
assert SPEC is not None and SPEC.loader is not None
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


def _enriched_rows() -> pd.DataFrame:
    rows = []
    for method in ("npe_full_data", "npe_zero_imputation"):
        rows.append({
            "problem": "toy", "missingness": "none" if method == "npe_full_data" else "mnar",
            "epsilon": 0.0 if method == "npe_full_data" else 0.5,
            "method": method, "budget": "low_sim_budget", "seed": 101,
            "reference_c2st_accuracy_mean": 0.6, "reference_c2st_accuracy_count": 10,
            "reference_posterior_mean_shift_median": 0.2,
            "reference_posterior_mean_shift_count": 10,
            "reference_covariance_trace_ratio_median": 1.1,
            "reference_covariance_trace_ratio_count": 10,
            "tarp_mae": 0.03, "training_time_sec": 4.0, "diagnostics_status": "ok",
        })
    return pd.DataFrame(rows)


def _partial_rows() -> pd.DataFrame:
    rows = []
    for metric, value in (("c2st", 0.8), ("mean_shift", 0.4), ("covariance_trace_ratio", 2.0)):
        rows.append({
            "problem": "toy", "mechanism": "mnar", "budget": "low", "method": "npe_zero_imputation",
            "fraction": 0.5, "seed": 101, "metric": metric, "target": "partial", "value": value,
            "configuration_complete": False, "n_valid_reference_observations": 8,
            "source": "partial.json", "partial_original_reference_indices": "[0,1,3,4,5,6,8,9]",
            "reference_subset_mismatch": True,
        })
    return pd.DataFrame(rows)


def test_partial_overlay_changes_only_reference_metrics_and_preserves_sparse_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "enriched.csv"
    _enriched_rows().to_csv(source, index=False)
    monkeypatch.setattr(ADAPTER, "PARTIAL_CAMPAIGNS", (("toy", "mnar"),))
    monkeypatch.setattr(ADAPTER, "load_partial_seed_metrics", lambda root: ADAPTER._normalize(_partial_rows().rename(columns={"mechanism": "missingness", "fraction": "epsilon"})))

    inputs = ADAPTER.build_plot_inputs(source, tmp_path)
    full = inputs.full.set_index("method")
    partial = inputs.partial.set_index("method")
    assert partial.loc["npe_zero_imputation", "reference_c2st_accuracy_mean"] == pytest.approx(0.8)
    assert partial.loc["npe_zero_imputation", "reference_posterior_mean_shift_median"] == pytest.approx(0.4)
    assert partial.loc["npe_zero_imputation", "reference_covariance_trace_ratio_median"] == pytest.approx(2.0)
    assert partial.loc["npe_zero_imputation", "reference_c2st_accuracy_count"] == 8
    assert partial.loc["npe_zero_imputation", "partial_reference_subset_mismatch"]
    assert partial.loc["npe_zero_imputation", "partial_reference_original_indices"] == "[0,1,3,4,5,6,8,9]"
    for column in ("tarp_mae", "training_time_sec", "diagnostics_status"):
        assert partial.loc["npe_zero_imputation", column] == full.loc["npe_zero_imputation", column]
    for column in ADAPTER.REFERENCE_METRICS.values():
        assert partial.loc["npe_full_data", column[0]] == full.loc["npe_full_data", column[0]]


def test_strict_partial_join_rejects_missing_metric_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "enriched.csv"
    _enriched_rows().to_csv(source, index=False)
    monkeypatch.setattr(ADAPTER, "PARTIAL_CAMPAIGNS", (("toy", "mnar"),))
    partial = _partial_rows().iloc[:2].copy()
    monkeypatch.setattr(ADAPTER, "load_partial_seed_metrics", lambda root: ADAPTER._normalize(partial.rename(columns={"mechanism": "missingness", "fraction": "epsilon"})))
    with pytest.raises(ValueError, match="strictly cover"):
        ADAPTER.build_plot_inputs(source, tmp_path)


def test_comparison_representation_contains_matching_full_and_partial_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "enriched.csv"
    _enriched_rows().to_csv(source, index=False)
    monkeypatch.setattr(ADAPTER, "PARTIAL_CAMPAIGNS", (("toy", "mnar"),))
    monkeypatch.setattr(ADAPTER, "load_partial_seed_metrics", lambda root: ADAPTER._normalize(_partial_rows().rename(columns={"mechanism": "missingness", "fraction": "epsilon"})))
    comparison = ADAPTER.build_plot_inputs(source, tmp_path).comparison
    assert set(comparison.reference_target) == {"full", "partial"}
    assert len(comparison) == 4
    assert np.all(comparison.groupby("reference_target").size().to_numpy() == 2)
