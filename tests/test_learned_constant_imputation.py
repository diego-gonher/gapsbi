from __future__ import annotations

import glob
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from gapsbi.methods.learned_constant_imputation import LearnedConstantImputer


def test_learned_constant_imputer_replaces_only_missing_entries() -> None:
    imputer = LearnedConstantImputer(x_dim=3, init_value=-2.0)
    with torch.no_grad():
        imputer.constants.copy_(torch.tensor([1.0, 2.0, 3.0]))

    x = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    completed = imputer(x, mask)

    expected = torch.tensor([[10.0, 2.0, 30.0], [1.0, 50.0, 3.0]])
    assert torch.allclose(completed, expected)


def test_learned_constant_configs_match_benchmark_grid() -> None:
    paths = sorted(
        glob.glob(
            "experiments/npe_learned_constant_imputation/*_sim_budget/*/*.yaml"
        )
    )
    assert len(paths) == 108

    problems = set()
    budgets = set()
    missingness = set()
    epsilons = set()
    seeds = None

    for path in paths:
        config = yaml.safe_load(Path(path).read_text())
        parts = Path(path).parts
        budget = parts[2]
        problem_dir = parts[3]
        stem = Path(path).stem
        mechanism = stem.split("_eps")[0].split("_")[-1]
        eps = stem.split("_eps")[-1].replace("_config", "")
        budgets.add(budget)
        problems.add(config["problem"])
        missingness.add(mechanism)
        epsilons.add(eps)
        seeds = config["seeds"] if seeds is None else seeds

        assert problem_dir == config["problem"]
        assert config["output_dir"].startswith(
            f"outputs_{budget.replace('_sim_budget', '')}_sim_budget/"
            "npe_learned_constant_imputation/"
        )
        assert config["reference_path"] == (
            f"references/reference_posteriors_v1/{config['problem']}_references.h5"
        )
        assert config["learned_constant_imputation"]["init_value"] == 0.0
        assert config["npe"]["density_estimator"] == "nsf"

    assert problems == {"oup", "glm", "glu", "lotka_volterra"}
    assert budgets == {"low_sim_budget", "mid_sim_budget", "high_sim_budget"}
    assert missingness == {"mcar", "mar", "mnar"}
    assert epsilons == {"010", "025", "050"}
    assert seeds == [101, 202, 303, 404, 505]


def test_learned_constant_reference_h5_contract(tmp_path: Path) -> None:
    path = tmp_path / "reference_posterior_samples.h5"
    constants = np.array([0.1, -0.2])
    with h5py.File(path, "w") as f:
        f.create_dataset("learned_constants_scaled", data=constants)
        f.attrs["reference_metric_target"] = (
            "masked_learned_constant_x_vs_full_x_reference"
        )

    with h5py.File(path, "r") as f:
        np.testing.assert_allclose(f["learned_constants_scaled"][...], constants)
        assert (
            f.attrs["reference_metric_target"]
            == "masked_learned_constant_x_vs_full_x_reference"
        )
