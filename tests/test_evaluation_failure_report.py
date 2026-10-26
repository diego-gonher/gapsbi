"""Regression coverage for nonfinite-evaluation reliability reporting."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


SPEC = importlib.util.spec_from_file_location(
    "report_evaluation_failures", Path(__file__).parents[1] / "scripts/report_evaluation_failures.py"
)
assert SPEC is not None and SPEC.loader is not None
REPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPORT
SPEC.loader.exec_module(REPORT)


def test_failure_report_separates_sparse_references_from_nonfinite_estimates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(REPORT, "CAMPAIGNS", (("lotka_volterra", "mnar"),))
    directory = tmp_path / "lotka_volterra_mnar"
    directory.mkdir()
    rows = [
        {"problem": "lotka_volterra", "mechanism": "mnar", "budget": "low", "method": "method",
         "fraction": .5, "seed": 101, "metric": metric, "target": "partial",
         "n_valid_reference_observations": 7, "source": "saved.json"}
        for metric in REPORT.METRICS
    ]
    pd.DataFrame(rows).to_csv(directory / "seed_level_metrics.csv", index=False)

    details = REPORT.load_seed_details(tmp_path)
    assert details.loc[0, "available_reference_count"] == 8
    assert details.loc[0, "missing_reference_posterior_count"] == 2
    assert details.loc[0, "nonfinite_estimator_reference_count"] == 1
    summary = REPORT.summarize(details)
    assert summary.loc[0, "valid_reference_count_total"] == 7
    assert summary.loc[0, "nonfinite_estimator_reference_count_total"] == 1
