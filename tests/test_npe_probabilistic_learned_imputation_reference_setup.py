from pathlib import Path

import yaml


ROOT = Path("experiments/npe_probabilistic_learned_imputation")


def test_probabilistic_learned_imputation_config_grid_matches_benchmark_v1() -> None:
    configs = sorted(ROOT.glob("*_sim_budget/*/*.yaml"))
    assert len(configs) == 108

    budgets = {path.parts[2] for path in configs}
    problems = {path.parts[3] for path in configs}
    assert budgets == {"low_sim_budget", "mid_sim_budget", "high_sim_budget"}
    assert problems == {"oup", "glm", "glu", "lotka_volterra"}

    for path in configs:
        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        problem = path.parts[3]
        mechanism = path.stem.split("_eps")[0].split("_")[-1]
        assert config["problem"] == problem
        assert config["reference_path"] == (
            f"references/reference_posteriors_v1/{problem}_references.h5"
        )
        assert config["reference_num_posterior_samples"] == 10_000
        assert config["reference_metrics"]["c2st_n_folds"] == 3
        assert config["seeds"] == [101, 202, 303, 404, 505]
        assert "ricker" not in config["dataset_path"]
        assert "npe_rise" not in config["output_dir"]
        if mechanism == "mnar":
            assert config["rise"]["lambda_np"] == 1.0
        else:
            assert config["rise"]["lambda_np"] == 100.0
