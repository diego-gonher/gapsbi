import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gapsbi.datasets import generate_dataset, generate_split
from gapsbi.io import load_gapsbi_hdf5, save_gapsbi_hdf5, validate_gapsbi_dataset
from gapsbi.masks import BlockMCARMask, PointMCARMask
from gapsbi.rng import make_rng
from gapsbi.simulators import RickerSimulator


def test_generate_split_shapes() -> None:
    simulator = RickerSimulator(T=12)
    mask_generator = PointMCARMask(missing_fraction=0.25)

    split = generate_split(simulator, mask_generator, n=5, rng=make_rng(123))

    assert split["theta"].shape == (5, 2)
    assert split["x_full"].shape == (5, 12)
    assert split["x_obs"].shape == (5, 12)
    assert split["mask"].shape == (5, 12)
    np.testing.assert_array_equal(split["x_obs"], split["x_full"] * split["mask"])


def test_generate_dataset_splits() -> None:
    simulator = RickerSimulator(T=12)
    mask_generator = BlockMCARMask(missing_fraction=0.25, block_size=3)

    dataset = generate_dataset(
        simulator,
        mask_generator,
        n_train=5,
        n_val=3,
        n_test=2,
        seed=123,
    )

    assert set(dataset) == {"train", "val", "test"}
    assert dataset["train"]["theta"].shape == (5, 2)
    assert dataset["val"]["theta"].shape == (3, 2)
    assert dataset["test"]["theta"].shape == (2, 2)
    validate_gapsbi_dataset(dataset)


def test_save_load_roundtrip(tmp_path) -> None:
    simulator = RickerSimulator(T=12)
    mask_generator = PointMCARMask(missing_fraction=0.25)
    dataset = generate_dataset(simulator, mask_generator, 5, 3, 2, seed=123)
    metadata = {
        "task": "ricker",
        "seed": 123,
        "simulator": simulator.metadata(),
        "mask": mask_generator.metadata(),
    }
    path = tmp_path / "dataset.h5"

    save_gapsbi_hdf5(path, dataset, metadata=metadata)
    loaded_dataset, loaded_metadata = load_gapsbi_hdf5(path)

    for split in ("train", "val", "test"):
        for name in ("theta", "x_full", "x_obs", "mask"):
            np.testing.assert_array_equal(loaded_dataset[split][name], dataset[split][name])
    assert loaded_metadata["task"] == "ricker"
    assert loaded_metadata["seed"] == 123
    assert loaded_metadata["simulator"]["name"] == "ricker"
    assert loaded_metadata["mask"]["name"] == "point_mcar"


def test_validate_rejects_missing_split() -> None:
    dataset = generate_dataset(
        RickerSimulator(T=12),
        PointMCARMask(missing_fraction=0.25),
        5,
        3,
        2,
        seed=123,
    )
    dataset.pop("test")

    with pytest.raises(ValueError, match="Missing required split"):
        validate_gapsbi_dataset(dataset)


def test_validate_rejects_nonbinary_mask() -> None:
    dataset = generate_dataset(
        RickerSimulator(T=12),
        PointMCARMask(missing_fraction=0.25),
        5,
        3,
        2,
        seed=123,
    )
    dataset["train"]["mask"][0, 0] = 2

    with pytest.raises(ValueError, match="mask must be binary"):
        validate_gapsbi_dataset(dataset)


def test_validate_rejects_inconsistent_x_obs() -> None:
    dataset = generate_dataset(
        RickerSimulator(T=12),
        PointMCARMask(missing_fraction=0.25),
        5,
        3,
        2,
        seed=123,
    )
    dataset["train"]["x_obs"][0, 0] += 1

    with pytest.raises(ValueError, match="x_obs must equal"):
        validate_gapsbi_dataset(dataset)


def test_save_refuses_overwrite_by_default(tmp_path) -> None:
    dataset = generate_dataset(
        RickerSimulator(T=12),
        PointMCARMask(missing_fraction=0.25),
        5,
        3,
        2,
        seed=123,
    )
    path = tmp_path / "dataset.h5"

    save_gapsbi_hdf5(path, dataset)

    with pytest.raises(FileExistsError):
        save_gapsbi_hdf5(path, dataset)


def test_generate_dataset_cli_supports_ricker_log1p_mnar(tmp_path) -> None:
    output_path = tmp_path / "ricker_log1p_mnar.h5"
    repo_root = Path(__file__).parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_dataset.py",
            "--task",
            "ricker",
            "--mask",
            "self_censoring_mnar",
            "--missing-fraction",
            "1.0",
            "--mnar-score-transform",
            "log1p",
            "--n-train",
            "2",
            "--n-val",
            "1",
            "--n-test",
            "1",
            "--seed",
            "123",
            "--output",
            str(output_path),
        ],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert output_path.exists()
    assert "Saved:" in result.stdout
    _, metadata = load_gapsbi_hdf5(output_path)
    assert metadata["mask"]["name"] == "self_censoring_mnar"
    assert metadata["mask"]["score_transform"] == "log1p"
