from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_full_data_moment_shift_module():
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    script_path = scripts_dir / "compute_full_data_per_observation_seed_moment_shift_metrics.py"
    spec = importlib.util.spec_from_file_location(
        "compute_full_data_per_observation_seed_moment_shift_metrics",
        script_path,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_error_row_preserves_metadata_and_status() -> None:
    metrics = _load_full_data_moment_shift_module()
    base = {
        "problem": "oup",
        "seed_a": "1",
        "seed_b": "2",
        "run_dir_a": "a",
        "run_dir_b": "b",
    }

    row = metrics.error_row(base, observation_index=3, error="bad covariance")

    assert row["problem"] == "oup"
    assert row["observation_index"] == 3
    assert row["status"] == "error"
    assert row["error"] == "bad covariance"


def test_build_summary_rows_groups_ok_rows_by_problem() -> None:
    metrics = _load_full_data_moment_shift_module()
    rows = [
        {
            "problem": "oup",
            "euclidean_mean_shift": 1.0,
            "mahalanobis_mean_shift": 2.0,
            "cov_trace_ratio": 1.2,
            "logdet_cov_ratio": 0.3,
            "status": "ok",
        },
        {
            "problem": "oup",
            "euclidean_mean_shift": 3.0,
            "mahalanobis_mean_shift": 4.0,
            "cov_trace_ratio": 1.4,
            "logdet_cov_ratio": 0.5,
            "status": "ok",
        },
        {
            "problem": "glm",
            "euclidean_mean_shift": "",
            "mahalanobis_mean_shift": "",
            "cov_trace_ratio": "",
            "logdet_cov_ratio": "",
            "status": "error",
        },
    ]

    summary = metrics.build_summary_rows(rows)

    assert summary[0]["problem"] == "glm"
    assert summary[0]["count"] == 0
    assert summary[1]["problem"] == "oup"
    assert summary[1]["count"] == 2
    assert abs(summary[1]["euclidean_mean_shift_mean"] - 2.0) < 1e-12
    assert abs(summary[1]["mahalanobis_mean_shift_median"] - 3.0) < 1e-12
    assert abs(summary[1]["cov_trace_ratio_min"] - 1.2) < 1e-12
    assert abs(summary[1]["logdet_cov_ratio_max"] - 0.5) < 1e-12
