from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_full_data_shift_module():
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    script_path = scripts_dir / "compute_full_data_seed_shift_metrics.py"
    spec = importlib.util.spec_from_file_location("compute_full_data_seed_shift_metrics", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_full_data_seed_pairs_groups_by_problem() -> None:
    metrics = _load_full_data_shift_module()
    rows = [
        {"method": "full_data", "problem": "oup", "seed": "3", "run_dir": "oup/3"},
        {"method": "full_data", "problem": "oup", "seed": "1", "run_dir": "oup/1"},
        {"method": "full_data", "problem": "oup", "seed": "2", "run_dir": "oup/2"},
        {"method": "full_data", "problem": "glm", "seed": "2", "run_dir": "glm/2"},
        {"method": "full_data", "problem": "glm", "seed": "1", "run_dir": "glm/1"},
    ]

    pairs = metrics.build_full_data_seed_pairs(rows)

    assert [(pair[0]["problem"], pair[0]["seed"], pair[1]["seed"]) for pair in pairs] == [
        ("glm", "1", "2"),
        ("oup", "1", "2"),
        ("oup", "1", "3"),
        ("oup", "2", "3"),
    ]


def test_build_summary_rows_groups_by_problem() -> None:
    metrics = _load_full_data_shift_module()
    rows = [
        {"problem": "oup", "seed_a": "1", "seed_b": "2", "mmd2_rbf": 0.1, "c2st_accuracy": 0.55, "status": "ok"},
        {"problem": "oup", "seed_a": "1", "seed_b": "3", "mmd2_rbf": 0.3, "c2st_accuracy": 0.65, "status": "ok"},
        {"problem": "glm", "seed_a": "1", "seed_b": "2", "mmd2_rbf": "", "c2st_accuracy": "", "status": "error"},
    ]

    summary_rows = metrics.build_summary_rows(rows)

    assert summary_rows[0]["problem"] == "glm"
    assert summary_rows[0]["num_pairs"] == 1
    assert summary_rows[1]["problem"] == "oup"
    assert summary_rows[1]["num_pairs"] == 2
    assert abs(summary_rows[1]["mmd2_rbf_mean"] - 0.2) < 1e-12
    assert abs(summary_rows[1]["c2st_accuracy_mean"] - 0.6) < 1e-12
