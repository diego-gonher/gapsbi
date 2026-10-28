"""Release-facing defaults for benchmark plotting."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("plot_benchmark_summary", SCRIPTS / "plot_benchmark_summary.py")
assert SPEC is not None and SPEC.loader is not None
PLOTTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLOTTER
SPEC.loader.exec_module(PLOTTER)


def test_release_plot_defaults_use_partial_mcar_mnar_only(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["plot_benchmark_summary.py", "--all"])
    args = PLOTTER.parse_args()
    assert args.reference_mode == "partial"
    assert {mechanism for _, mechanism in PLOTTER.selected_jobs(args)} == {"mcar", "mnar"}


def test_nre_c_overlay_loader_requires_complete_seed_grid_and_maps_budgets(tmp_path: Path) -> None:
    methods = sorted(PLOTTER.NRE_C_METHODS.values())
    seeds = [101, 202, 303, 404, 505]
    budgets = ["low", "mid", "high"]

    for mechanism in ("mcar", "mnar"):
        directory = tmp_path / f"oup_{mechanism}_eps010"
        directory.mkdir()
        rows = []
        for method_i, method in enumerate(methods):
            for budget_i, budget in enumerate(budgets):
                for seed_i, seed in enumerate(seeds):
                    rows.append({
                        "backbone": "NRE-C", "target": "partial", "problem": "oup",
                        "mechanism": mechanism, "fraction": 0.10, "method": method,
                        "budget": budget, "seed": seed, "metric": "c2st",
                        "value": 0.6 + method_i * 0.01 + budget_i * 0.001 + seed_i * 0.0001,
                        "configuration_complete": True,
                        "n_valid_reference_observations": 10,
                    })
        if mechanism == "mcar":
            for budget_i, budget in enumerate(budgets):
                for seed_i, seed in enumerate(seeds):
                    rows.append({
                        "backbone": "NRE-C", "target": "full", "problem": "oup",
                        "mechanism": mechanism, "fraction": 0.10, "method": "nre_c_full_data",
                        "budget": budget, "seed": seed, "metric": "c2st",
                        "value": 0.7 + budget_i * 0.001 + seed_i * 0.0001,
                        "configuration_complete": True,
                        "n_valid_reference_observations": 10,
                    })
        seed_frame = pd.DataFrame(rows)
        seed_frame.to_csv(directory / "comparison_seed_level_metrics.csv", index=False)
        partial = seed_frame[seed_frame["target"] == "partial"]
        aggregate = partial.groupby(["backbone", "target", "metric", "method", "budget"], as_index=False).agg(
            mean=("value", "mean"), std=("value", "std"), n_seeds=("value", "count")
        )
        aggregate.to_csv(directory / "comparison_aggregate_metrics.csv", index=False)

    overlay = PLOTTER.load_nre_c_oup_overlay(tmp_path)
    assert len(overlay) == 165
    assert set(overlay["budget"]) == set(PLOTTER.BUDGET_ORDER)
    panel = PLOTTER.prepare_nre_c_overlay_panel(
        overlay, method=methods[0], mechanism="mcar"
    )
    assert panel["n"].tolist() == [5, 5, 5]
    assert np.isfinite(panel["ci95"]).all()
    assert PLOTTER.combined_legend_handles(include_nre_c=True)[-1].get_label() == "NRE-C (OUP, 10%)"
