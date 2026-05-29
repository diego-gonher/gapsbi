from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


def _load_analysis_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_campaign1_stability.py"
    spec = importlib.util.spec_from_file_location("analyze_campaign1_stability", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_master_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_analysis_builds_config_summary_and_report(tmp_path: Path) -> None:
    analysis = _load_analysis_module()
    master_path = tmp_path / "campaign1_master_results.csv"
    fieldnames = [
        "method",
        "problem",
        "missingness",
        "epsilon",
        "experiment",
        "seed",
        "run_dir",
        "status",
        *analysis.KEY_METRICS,
    ]
    rows = [
        {
            "method": "zero_imputation",
            "problem": "oup",
            "missingness": "mcar",
            "epsilon": "0.1",
            "experiment": "oup_mcar_eps010",
            "seed": seed,
            "run_dir": f"outputs/example/seed_{seed}",
            "status": "ok",
            "tarp_mae": 0.1 * idx,
            "tarp_iae": 0.2 * idx,
            "sbc_ks_pval_min": 0.5,
            "sbc_ks_pval_mean": 0.7,
            "training_time_sec": 10.0,
            "posterior_sampling_time_sec": 2.0,
            "diagnostics_time_sec": 1.0,
            "total_runtime_sec": 13.0,
        }
        for idx, seed in enumerate([1, 2, 3], start=1)
    ]
    _write_master_csv(master_path, rows, fieldnames)

    loaded = analysis.load_master_csv(master_path)
    outdir = tmp_path / "analysis"
    config_summary_path, method_summary_path, report_path = analysis.write_outputs(loaded, outdir)

    config_rows = list(csv.DictReader(config_summary_path.open(encoding="utf-8")))
    method_rows = list(csv.DictReader(method_summary_path.open(encoding="utf-8")))
    report = report_path.read_text(encoding="utf-8")

    assert len(config_rows) == 1
    assert len(method_rows) == 1
    assert abs(float(config_rows[0]["tarp_mae_mean"]) - 0.2) < 1e-12
    assert config_rows[0]["tarp_mae_count"] == "3"
    assert "Configs with fewer/more than 5 seeds: 1" in report
    assert "Top 10 worst configs by mean tarp_iae" in report
