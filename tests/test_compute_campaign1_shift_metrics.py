from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


def _load_shift_metrics_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "compute_campaign1_shift_metrics.py"
    spec = importlib.util.spec_from_file_location("compute_campaign1_shift_metrics", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_posterior(run_dir: Path, value: float) -> None:
    run_dir.mkdir(parents=True)
    samples = np.full((2, 6, 1), value, dtype=np.float32)
    with h5py.File(run_dir / "posterior_samples.h5", "w") as h5:
        h5.create_dataset("theta_posterior_scaled", data=samples)


def test_mmd_is_near_zero_for_identical_samples() -> None:
    metrics = _load_shift_metrics_module()
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(80, 3))

    mmd2 = metrics.rbf_mmd2(samples, samples.copy())

    assert mmd2 < 1e-12


def test_mmd_is_larger_for_shifted_samples() -> None:
    metrics = _load_shift_metrics_module()
    rng = np.random.default_rng(1)
    samples = rng.normal(size=(120, 2))
    shifted = samples + 2.0

    same_mmd2 = metrics.rbf_mmd2(samples, samples.copy())
    shifted_mmd2 = metrics.rbf_mmd2(samples, shifted)

    assert shifted_mmd2 > same_mmd2 + 0.1


def test_c2st_is_near_chance_for_same_distribution() -> None:
    metrics = _load_shift_metrics_module()
    rng = np.random.default_rng(2)
    x = rng.normal(size=(300, 4))
    y = rng.normal(size=(300, 4))

    accuracy = metrics.c2st_accuracy(x, y, seed=2)

    assert 0.35 <= accuracy <= 0.65


def test_c2st_is_high_for_clearly_separated_samples() -> None:
    metrics = _load_shift_metrics_module()
    rng = np.random.default_rng(3)
    x = rng.normal(size=(300, 4))
    y = rng.normal(loc=4.0, size=(300, 4))

    accuracy = metrics.c2st_accuracy(x, y, seed=3)

    assert accuracy > 0.9


def test_select_posterior_key_prefers_scaled(tmp_path: Path) -> None:
    metrics = _load_shift_metrics_module()
    path = tmp_path / "posterior_samples.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("theta_posterior", data=np.zeros((2, 3, 1), dtype=np.float32))
        h5.create_dataset("theta_posterior_scaled", data=np.ones((2, 3, 1), dtype=np.float32))

    with h5py.File(path, "r") as h5:
        key = metrics.select_posterior_key(h5)

    assert key == "theta_posterior_scaled"


def test_default_same_input_reference_behavior_has_original_schema(tmp_path: Path) -> None:
    metrics = _load_shift_metrics_module()
    missing_dir = tmp_path / "missing"
    full_dir = tmp_path / "full"
    _write_posterior(missing_dir, 1.0)
    _write_posterior(full_dir, 0.0)
    rows = [
        {
            "method": "full_data",
            "problem": "oup",
            "seed": "1",
            "run_dir": str(full_dir),
        },
        {
            "method": "npe_missing",
            "problem": "oup",
            "missingness": "mcar",
            "epsilon": "0.5",
            "experiment": "oup_mcar",
            "seed": "1",
            "run_dir": str(missing_dir),
        },
    ]

    result = metrics.build_shift_metric_rows(rows, max_samples=4, seed=0, c2st_test_size=0.5)

    assert len(result) == 1
    assert result[0]["status"] == "ok"
    assert result[0]["full_run_dir"] == str(full_dir)
    assert "reference_input" not in result[0]


def test_cross_input_reference_matching_uses_reference_csv_rows(tmp_path: Path) -> None:
    metrics = _load_shift_metrics_module()
    missing_dir = tmp_path / "missing"
    input_full_dir = tmp_path / "input_full"
    reference_full_dir = tmp_path / "reference_full"
    _write_posterior(missing_dir, 1.0)
    _write_posterior(input_full_dir, 100.0)
    _write_posterior(reference_full_dir, 0.0)
    rows = [
        {
            "method": "full_data",
            "problem": "oup",
            "seed": "1",
            "run_dir": str(input_full_dir),
        },
        {
            "method": "npe_missing",
            "problem": "oup",
            "missingness": "mcar",
            "epsilon": "0.5",
            "experiment": "oup_mcar",
            "seed": "1",
            "run_dir": str(missing_dir),
        },
    ]
    reference_rows = [
        {
            "method": "full_data",
            "problem": "oup",
            "seed": "1",
            "run_dir": str(reference_full_dir),
            "simulation_budget": "90000",
        }
    ]
    reference_input = tmp_path / "reference_master.csv"

    result = metrics.build_shift_metric_rows(
        rows,
        reference_rows=reference_rows,
        reference_input=reference_input,
        max_samples=4,
        seed=0,
        c2st_test_size=0.5,
    )

    assert len(result) == 1
    assert result[0]["status"] == "ok"
    assert result[0]["full_run_dir"] == str(reference_full_dir)
    assert result[0]["reference_input"] == str(reference_input)
    assert result[0]["reference_run_dir"] == str(reference_full_dir)
    assert result[0]["reference_budget"] == "90000"


def test_cross_input_ignores_full_data_rows_from_input(tmp_path: Path) -> None:
    metrics = _load_shift_metrics_module()
    missing_dir = tmp_path / "missing"
    input_full_dir = tmp_path / "input_full"
    reference_full_dir = tmp_path / "reference_full"
    _write_posterior(missing_dir, 1.0)
    _write_posterior(input_full_dir, 100.0)
    _write_posterior(reference_full_dir, 0.0)
    rows = [
        {"method": "full_data", "problem": "oup", "seed": "1", "run_dir": str(input_full_dir)},
        {
            "method": "npe_missing",
            "problem": "oup",
            "missingness": "mcar",
            "epsilon": "0.5",
            "experiment": "oup_mcar",
            "seed": "1",
            "run_dir": str(missing_dir),
        },
    ]
    reference_rows = [{"method": "full_data", "problem": "oup", "seed": "1", "run_dir": str(reference_full_dir)}]

    missing_rows, full_rows = metrics.prepare_reference_rows(rows, reference_rows=reference_rows)

    assert [row["method"] for row in missing_rows] == ["npe_missing"]
    assert full_rows[("oup", "1")]["run_dir"] == str(reference_full_dir)


def test_missing_cross_input_reference_row_fails_gracefully(tmp_path: Path) -> None:
    metrics = _load_shift_metrics_module()
    missing_dir = tmp_path / "missing"
    _write_posterior(missing_dir, 1.0)
    rows = [
        {
            "method": "npe_missing",
            "problem": "oup",
            "missingness": "mcar",
            "epsilon": "0.5",
            "experiment": "oup_mcar",
            "seed": "1",
            "run_dir": str(missing_dir),
        }
    ]

    result = metrics.build_shift_metric_rows(
        rows,
        reference_rows=[],
        reference_input=tmp_path / "reference_master.csv",
        max_samples=4,
        seed=0,
        c2st_test_size=0.5,
    )

    assert len(result) == 1
    assert result[0]["status"] == "error"
    assert "Missing full-data reference row for problem=oup seed=1" in result[0]["error"]


@pytest.mark.parametrize(
    ("module_name", "script_name"),
    [
        ("compute_campaign1_shift_metrics", "compute_campaign1_shift_metrics.py"),
        ("compute_campaign1_per_observation_shift_metrics", "compute_campaign1_per_observation_shift_metrics.py"),
        (
            "compute_campaign1_per_observation_moment_shift_metrics",
            "compute_campaign1_per_observation_moment_shift_metrics.py",
        ),
    ],
)
def test_modified_scripts_accept_reference_input_cli(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    script_name: str,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    script_path = scripts_dir / script_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        sys,
        "argv",
        [script_name, "--input", "input.csv", "--reference-input", "reference.csv"],
    )

    args = module.parse_args()

    assert args.reference_input == Path("reference.csv")
