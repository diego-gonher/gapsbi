from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_aggregator_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_campaign1_results.py"
    spec = importlib.util.spec_from_file_location("aggregate_campaign1_results", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_summary(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_parse_missing_data_result_path(tmp_path: Path) -> None:
    aggregator = _load_aggregator_module()
    outputs_dir = tmp_path / "outputs"
    summary_path = (
        outputs_dir
        / "npe_mask_augmentation"
        / "ricker"
        / "mnar"
        / "ricker_mnar_eps025"
        / "seed_26485"
        / "summary.json"
    )
    _write_summary(summary_path, {"seed": 26485, "problem": "ricker"})

    parsed = aggregator.parse_result_path(summary_path, outputs_dir)

    assert parsed["method"] == "mean_imputation_mask"
    assert parsed["problem"] == "ricker"
    assert parsed["missingness"] == "mnar"
    assert parsed["epsilon"] == 0.25
    assert parsed["experiment"] == "ricker_mnar_eps025"
    assert parsed["seed"] == 26485


def test_parse_full_data_result_path(tmp_path: Path) -> None:
    aggregator = _load_aggregator_module()
    outputs_dir = tmp_path / "outputs"
    summary_path = outputs_dir / "npe_full_data" / "oup_full_data" / "seed_976532" / "summary.json"
    _write_summary(summary_path, {"seed": 976532, "problem": "oup"})

    parsed = aggregator.parse_result_path(summary_path, outputs_dir)

    assert parsed["method"] == "full_data"
    assert parsed["problem"] == "oup"
    assert parsed["missingness"] == "none"
    assert parsed["epsilon"] == 0.0
    assert parsed["experiment"] == "oup_full_data"
    assert parsed["seed"] == 976532


def test_build_results_table_fills_missing_metrics_with_nan(tmp_path: Path) -> None:
    aggregator = _load_aggregator_module()
    outputs_dir = tmp_path / "outputs"
    summary_path = (
        outputs_dir
        / "npe_zero_imputation"
        / "glm"
        / "mcar"
        / "glm_raw_mcar_eps010"
        / "seed_1172401"
        / "summary.json"
    )
    _write_summary(
        summary_path,
        {
            "seed": 1172401,
            "problem": "glm",
            "training_time_sec": 1.0,
            "posterior_sampling_time_sec": 2.0,
            "diagnostics_time_sec": 3.0,
            "total_runtime_sec": 6.0,
        },
    )

    rows = aggregator.build_results_table(outputs_dir)

    assert len(rows) == 1
    row = rows[0]
    assert row["method"] == "zero_imputation"
    assert row["epsilon"] == 0.10
    assert row["status"] == "ok"
    assert row["training_time_sec"] == 1.0
    assert row["tarp_mae"] != row["tarp_mae"]
