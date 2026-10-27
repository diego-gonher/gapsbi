import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aggregate = _load("aggregate_nre_c_comparison", "scripts/aggregate_nre_c_comparison.py")
plotter = _load("plot_npe_nre_c_comparison", "scripts/plot_npe_nre_c_comparison.py")


def test_across_seed_mean_sample_sd_and_ci_are_exact():
    result = aggregate.aggregate_seed_values([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["mean"] == 3.0
    assert result["std"] == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1))
    assert result["ci95_low"] == pytest.approx(3.0 - 1.96 * result["std"] / np.sqrt(5))
    assert result["n_seeds"] == 5


def test_reference_id_alignment_rejects_missing_and_duplicate_ids():
    aggregate.validate_ids(list(range(10)), list(range(10)), "fixture")
    with pytest.raises(ValueError, match="reference IDs"):
        aggregate.validate_ids(list(range(9)), list(range(10)), "missing")
    with pytest.raises(ValueError, match="reference IDs"):
        aggregate.validate_ids([0, 0], [0, 1], "duplicate")


def test_configuration_grid_rejects_missing_and_duplicate_seed_metrics():
    frame = pd.DataFrame([
        {"method": method, "budget": budget, "seed": seed, "metric": metric}
        for method in ("npe_zero_imputation",)
        for budget in ("low",)
        for seed in (101, 202)
        for metric in ("c2st", "mean_shift", "covariance_trace_ratio")
    ])
    aggregate.validate_configuration_grid(frame, methods=["npe_zero_imputation"], budgets=["low"], seeds=[101, 202])
    with pytest.raises(ValueError, match="coverage mismatch"):
        aggregate.validate_configuration_grid(frame.iloc[:-1], methods=["npe_zero_imputation"], budgets=["low"], seeds=[101, 202])
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate.validate_configuration_grid(pd.concat([frame, frame.iloc[[0]]]), methods=["npe_zero_imputation"], budgets=["low"], seeds=[101, 202])


def test_existing_npe_aggregate_must_reproduce_from_seed_rows():
    values = [0.51, 0.54, 0.53, 0.52, 0.55]
    seed_frame = pd.DataFrame([
        {"method":"npe_zero_imputation","budget":"low","seed":seed,"metric":metric,"value":value + offset}
        for metric, offset in (("c2st",0.),("mean_shift",.1),("covariance_trace_ratio",1.))
        for seed, value in zip((101, 202, 303, 404, 505), values, strict=True)
    ])
    summary_rows=[]
    for metric,offset in (("c2st",0.),("mean_shift",.1),("covariance_trace_ratio",1.)):
        summary=aggregate.aggregate_seed_values([x+offset for x in values])
        summary_rows.append({"method":"npe_zero_imputation","budget":"low","metric":metric,"fraction":.1,
            "mean_partial":summary["mean"],"std_partial":summary["std"],"n_seeds_partial":5,"n_complete_configurations_partial":5})
    summary_frame = pd.DataFrame(summary_rows)
    aggregate._validate_npe_aggregates(seed_frame, summary_frame, ["npe_zero_imputation"], ["low"], [101,202,303,404,505])
    summary_frame.loc[0,"mean_partial"] += .01
    with pytest.raises(ValueError, match="does not reproduce"):
        aggregate._validate_npe_aggregates(seed_frame, summary_frame, ["npe_zero_imputation"], ["low"], [101,202,303,404,505])


def test_plot_panel_uses_saved_mean_and_seed_ci_values():
    rows=[]
    for backbone, prefix, shift in (("NPE","npe",0.0),("NRE-C","nre_c",.1)):
        for budget, value in zip(("low","mid","high"),(.7,.6,.55),strict=True):
            rows.append({"backbone":backbone,"method":f"{prefix}_zero_imputation","budget":budget,"target":"partial","metric":"c2st",
                "mean":value+shift,"std":.02,"ci95_low":value+shift-.01,"ci95_high":value+shift+.01,"n_seeds":5})
    frame=plotter.comparison_columns(pd.DataFrame(rows))
    series=plotter.panel_series(frame,"c2st","zero_imputation","NRE-C")
    assert series["mean"].tolist()==pytest.approx([.8,.7,.65])
    assert series["ci95_low"].tolist()==pytest.approx([.79,.69,.64])
    assert series["n_seeds"].tolist()==[5,5,5]

