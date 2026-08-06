from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent

SEEDS = [101, 202, 303, 404, 505]
PROBLEMS = ("oup", "glm", "glu", "lotka_volterra")
MECHANISMS = ("mcar", "mar", "mnar")
EPSILONS = ("010", "025", "050")

BUDGETS = {
    "high_sim_budget": {
        "output_prefix": "outputs_high_sim_budget",
        "training_batch_size": 256,
    },
    "mid_sim_budget": {
        "output_prefix": "outputs_mid_sim_budget",
        "max_train_samples": 9_000,
        "max_val_samples": 1_000,
        "training_batch_size": 256,
    },
    "low_sim_budget": {
        "output_prefix": "outputs_low_sim_budget",
        "max_train_samples": 900,
        "max_val_samples": 100,
        "training_batch_size": 100,
    },
}

DATASETS = {
    ("oup", "mcar"): "data/canonical_v1/oup/mcar/oup_mcar_eps{eps}_seed123.h5",
    ("oup", "mar"): "data/canonical_v1/oup/mar/oup_mar_coordinate_increasing_eps{eps}_seed123.h5",
    ("oup", "mnar"): (
        "data/canonical_v1/oup/mnar/"
        "oup_mnar_self_censoring_mean_normalized_identity_eps{eps}_seed123.h5"
    ),
    ("glm", "mcar"): "data/canonical_v1/glm/mcar/glm_raw_mcar_eps{eps}_seed123.h5",
    ("glm", "mar"): (
        "data/canonical_v1/glm/mar/"
        "glm_raw_mar_coordinate_increasing_eps{eps}_seed123.h5"
    ),
    ("glm", "mnar"): (
        "data/canonical_v1/glm/mnar/"
        "glm_raw_mnar_self_censoring_mean_normalized_identity_eps{eps}_seed123.h5"
    ),
    ("glu", "mcar"): "data/canonical_v1/glu/mcar/glu_mcar_eps{eps}_seed123.h5",
    ("glu", "mar"): "data/canonical_v1/glu/mar/glu_mar_coordinate_increasing_eps{eps}_seed123.h5",
    ("glu", "mnar"): (
        "data/canonical_v1/glu/mnar/"
        "glu_mnar_self_censoring_mean_normalized_identity_eps{eps}_seed123.h5"
    ),
    ("lotka_volterra", "mcar"): (
        "data/canonical_v1/lotka_volterra/mcar/"
        "lotka_volterra_time_block_mcar_eps{eps}_seed123.h5"
    ),
    ("lotka_volterra", "mar"): (
        "data/canonical_v1/lotka_volterra/mar/"
        "lotka_volterra_time_mar_increasing_eps{eps}_seed123.h5"
    ),
    ("lotka_volterra", "mnar"): (
        "data/canonical_v1/lotka_volterra/mnar/"
        "lotka_volterra_log_total_mnar_eps{eps}_seed123.h5"
    ),
}

OUTPUT_STEMS = {
    ("oup", "mcar"): "oup_mcar_eps{eps}",
    ("oup", "mar"): "oup_mar_eps{eps}",
    ("oup", "mnar"): "oup_mnar_eps{eps}",
    ("glm", "mcar"): "glm_raw_mcar_eps{eps}",
    ("glm", "mar"): "glm_raw_mar_eps{eps}",
    ("glm", "mnar"): "glm_raw_mnar_eps{eps}",
    ("glu", "mcar"): "glu_mcar_eps{eps}",
    ("glu", "mar"): "glu_mar_eps{eps}",
    ("glu", "mnar"): "glu_mnar_eps{eps}",
    ("lotka_volterra", "mcar"): "lotka_volterra_time_block_mcar_eps{eps}",
    ("lotka_volterra", "mar"): "lotka_volterra_time_mar_eps{eps}",
    ("lotka_volterra", "mnar"): "lotka_volterra_log_total_mnar_eps{eps}",
}


def config_for(*, problem: str, budget: str, mechanism: str, eps: str) -> dict:
    budget_cfg = BUDGETS[budget]
    output_stem = OUTPUT_STEMS[(problem, mechanism)].format(eps=eps)
    config = {
        "dataset_path": DATASETS[(problem, mechanism)].format(eps=eps),
        "problem": problem,
        "output_dir": (
            f"{budget_cfg['output_prefix']}/npe_learned_constant_imputation/"
            f"{problem}/{mechanism}/{output_stem}"
        ),
        "seeds": SEEDS,
        "reference_path": f"references/reference_posteriors_v1/{problem}_references.h5",
        "reference_mask_seed": 987_654,
        "reference_num_posterior_samples": 10_000,
        "reference_metrics": {
            "c2st_max_samples_per_observation": 2_000,
            "c2st_n_folds": 3,
            "c2st_hidden_layer_scale": 5,
        },
        "npe": {
            "density_estimator": "nsf",
            "device": "cpu",
            "training_batch_size": budget_cfg["training_batch_size"],
            "stop_after_epochs": 20,
            "max_num_epochs": 5_000,
        },
        "learned_constant_imputation": {
            "init_value": 0.0,
            "learning_rate": 1e-3,
            "constant_learning_rate": 1e-3,
        },
        "evaluation": {
            "test_sample": 1_000,
            "num_posterior_samples": 1_000,
            "num_alpha_grid": 101,
        },
        "sampling": {
            "reject_outside_prior": True,
            "max_sampling_time": 30.0,
        },
    }
    if "max_train_samples" in budget_cfg:
        config["max_train_samples"] = budget_cfg["max_train_samples"]
        config["max_val_samples"] = budget_cfg["max_val_samples"]
    return config


def write_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def main() -> None:
    written = 0
    for budget in BUDGETS:
        for problem in PROBLEMS:
            for mechanism in MECHANISMS:
                for eps in EPSILONS:
                    path = (
                        ROOT
                        / budget
                        / problem
                        / (
                            f"{problem}_npe_learned_constant_imputation_"
                            f"{mechanism}_eps{eps}_config.yaml"
                        )
                    )
                    write_config(
                        path,
                        config_for(
                            problem=problem,
                            budget=budget,
                            mechanism=mechanism,
                            eps=eps,
                        ),
                    )
                    written += 1
    print(f"Wrote {written} NPE learned-constant imputation configs under {ROOT}")


if __name__ == "__main__":
    main()
