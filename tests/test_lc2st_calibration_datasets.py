"""Tests for independent, test-only L-C2ST calibration data generation."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

from gapsbi.io import load_gapsbi_hdf5, validate_gapsbi_dataset


SPEC = importlib.util.spec_from_file_location(
    "generate_lc2st_calibration_datasets",
    Path(__file__).parents[1] / "scripts/generate_lc2st_calibration_datasets.py",
)
assert SPEC is not None and SPEC.loader is not None
GENERATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GENERATOR
SPEC.loader.exec_module(GENERATOR)


def test_calibration_generation_is_deterministic_and_test_only() -> None:
    first, _simulator, _mask = GENERATOR.generate_calibration_dataset("glu", 12, 31_301)
    second, _simulator, _mask = GENERATOR.generate_calibration_dataset("glu", 12, 31_301)

    validate_gapsbi_dataset(first)
    assert first["train"]["theta"].shape[0] == 0
    assert first["val"]["theta"].shape[0] == 0
    assert first["test"]["theta"].shape == (12, 10)
    for name in ("theta", "x_full", "x_obs", "mask"):
        np.testing.assert_array_equal(first["test"][name], second["test"][name])
    np.testing.assert_array_equal(
        first["test"]["x_obs"], first["test"]["x_full"] * first["test"]["mask"]
    )


def test_each_task_uses_canonical_shapes_and_mcar_masks() -> None:
    expected = {
        "oup": (2, 25), "glm": (10, 100), "glu": (10, 10), "lotka_volterra": (4, 100),
    }
    for task, (theta_dim, x_dim) in expected.items():
        dataset, _simulator, mask_generator = GENERATOR.generate_calibration_dataset(
            task, 8, GENERATOR.TASK_SEEDS[task]
        )
        test = dataset["test"]
        assert test["theta"].shape == (8, theta_dim)
        assert test["x_full"].shape == (8, x_dim)
        assert np.all((test["mask"] == 0) | (test["mask"] == 1))
        assert mask_generator.metadata()["missing_fraction"] == 0.10


def test_point_mcar_calibration_has_expected_observation_rate() -> None:
    dataset, _simulator, _mask = GENERATOR.generate_calibration_dataset("glu", 500, 31_301)
    observed_fraction = float(dataset["test"]["mask"].mean())
    assert 0.85 < observed_fraction < 0.95


def test_lv_mask_hides_complete_time_blocks() -> None:
    dataset, _simulator, _mask = GENERATOR.generate_calibration_dataset("lotka_volterra", 8, 31_401)
    mask = dataset["test"]["mask"].reshape(8, 50, 2)
    assert np.array_equal(mask[:, :, 0], mask[:, :, 1])
    assert np.all(np.sum(mask[:, :, 0] == 0, axis=1) == 5)


def test_calibration_seed_streams_do_not_overlap_canonical_style_draws() -> None:
    """Fixed calibration streams differ from canonical benchmark seed 123."""
    for task, seed in GENERATOR.TASK_SEEDS.items():
        calibration, _simulator, _mask = GENERATOR.generate_calibration_dataset(task, 8, seed)
        canonical_style, _simulator, _mask = GENERATOR.generate_calibration_dataset(task, 8, 123)
        calibration_rows = {
            tuple(np.concatenate((theta, x_full, mask)))
            for theta, x_full, mask in zip(
                calibration["test"]["theta"], calibration["test"]["x_full"], calibration["test"]["mask"], strict=True
            )
        }
        canonical_rows = {
            tuple(np.concatenate((theta, x_full, mask)))
            for theta, x_full, mask in zip(
                canonical_style["test"]["theta"], canonical_style["test"]["x_full"], canonical_style["test"]["mask"], strict=True
            )
        }
        assert calibration_rows.isdisjoint(canonical_rows)


def test_written_dataset_has_reproducibility_metadata_and_is_not_canonical(tmp_path: Path) -> None:
    output = GENERATOR.write_calibration_dataset(
        task="oup", num_calibration=9, seed=31_101, output_root=tmp_path,
        commit="test-commit", overwrite=False,
    )
    dataset, metadata = load_gapsbi_hdf5(output)

    assert "canonical_v1" not in str(output)
    assert metadata["calibration_role"] == GENERATOR.CALIBRATION_ROLE
    assert metadata["source_commit"] == "test-commit"
    assert metadata["mechanism"] == "mcar"
    assert metadata["missing_fraction"] == 0.10
    assert metadata["seed"] == 31_101
    assert dataset["test"]["theta"].shape[0] == 9
