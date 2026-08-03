# NPE Imputation Experiments

Naive imputation-only NPE baselines for the benchmark v1 missing-data tasks.

This experiment family trains NPE on `x_obs` after scaling and imputation, without
concatenating masks and without any mask-augmented architecture.

## Baselines

- `zero`: missing entries are replaced with `0.0` in scaled x-space.
- `mean`: missing entries are replaced with observed-only train feature means in
  scaled x-space.

Method names recorded in outputs:

- `npe_zero_imputation`
- `npe_mean_imputation`

## Canonical preprocessing

1. Load `theta`, `x_full`, `x_obs`, and `mask` from train/val/test HDF5 groups.
2. Scale `theta` with train-only `StandardScaler` for GLM/Lotka-Volterra and `MinMaxScaler(feature_range=(-1, 1))` for bounded-prior tasks.
3. Fit x scaler on `x_full_train` only:
   - GLM/GLU/OUP/Lotka-Volterra: `StandardScaler`
   - legacy Ricker: `log1p + StandardScaler`
4. Transform `x_obs` with that scaler.
5. Impute missing entries in scaled x-space (`zero` or `mean`).
6. Train `FixedSplitNPE_C` on imputed x only.

## Experiment Grid

Active configs cover:

- problems: OUP, GLM, GLU, Lotka-Volterra
- imputations: zero, mean
- budgets: low `900/100`, mid `9000/1000`, high full `90000/10000`
- missingness: MCAR, MAR, MNAR
- missing fractions: `0.10`, `0.25`, `0.50`
- seeds: `101, 202, 303, 404, 505`

Configs are generated from:

```bash
PYTHONPATH=src python experiments/npe_imputation/generate_configs.py
```

## Run Queues

```bash
bash experiments/npe_imputation/npe_imputation_run_queue_low_sim_budget.sh
bash experiments/npe_imputation/npe_imputation_run_queue_mid_sim_budget.sh
bash experiments/npe_imputation/npe_imputation_run_queue_high_sim_budget.sh
```

Each queue runs both zero and mean imputation for all problems, mechanisms, and
fractions.

## Single Run

```bash
PYTHONPATH=src python experiments/npe_imputation/train.py \
  --config experiments/npe_imputation/zero_imputation/low_sim_budget/glm/glm_zero_mcar_eps010_config.yaml
```

## Reference Metrics

Each config uses the 10 high-quality full-observation reference posteriors in
`references/reference_posteriors_v1`. For imputation baselines, the 10 reference
observations are deterministically masked with the dataset's missingness
mechanism, imputed in scaled x-space, and then compared against the full-data
reference posterior. The resulting metrics are a degradation-from-full-information
diagnostic, not the exact posterior under missingness.

Saved per-reference metrics:

- C2ST accuracy
- posterior mean shift
- covariance trace ratio

## Outputs

Each seed directory contains:

- `training_summary.png`
- `posterior_samples.h5`
- `tarp.png`
- `sbc_rank_histograms.png`
- `diagnostics_arrays.npz`
- `reference_posterior_samples.h5`
- `summary.json`

Experiment root contains:

- `all_results.json`
