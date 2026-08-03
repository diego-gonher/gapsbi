# NPE Mask-Augmentation Experiment

Wang-style mask augmentation baseline for the benchmark v1 missing-data tasks.

This experiment uses:

1. `x_obs` scaled with scaler fitted on `x_full_train`
2. zero imputation in scaled space
3. mask concatenation:
   `x_aug = [x_imputed, mask]`

Method recorded in outputs:

- `npe_mask_augmentation`

## Canonical preprocessing

1. Load `theta`, `x_full`, `x_obs`, `mask` for train/val/test.
2. Scale `theta` with train-only `StandardScaler` for GLM/Lotka-Volterra and `MinMaxScaler(-1, 1)` for bounded-prior tasks.
3. Fit x scaler on `x_full_train` only:
   - GLM/GLU/OUP/Lotka-Volterra: `StandardScaler`
   - legacy Ricker: `log1p + StandardScaler`
4. Transform `x_obs` using that fitted scaler.
5. Zero-impute missing entries in scaled space.
6. Concatenate `[x_imputed, mask.float()]` along feature axis.
7. Train `FixedSplitNPE_C` on augmented inputs.

## Experiment Grid

Active configs cover:

- problems: OUP, GLM, GLU, Lotka-Volterra
- budgets: low `900/100`, mid `9000/1000`, high full `90000/10000`
- missingness: MCAR, MAR, MNAR
- missing fractions: `0.10`, `0.25`, `0.50`
- seeds: `101, 202, 303, 404, 505`

Configs are generated from:

```bash
PYTHONPATH=src python experiments/npe_mask_augmentation/generate_configs.py
```

## Run Queues

```bash
bash experiments/npe_mask_augmentation/npe_mask_augmentation_run_queue_low_sim_budget.sh
bash experiments/npe_mask_augmentation/npe_mask_augmentation_run_queue_mid_sim_budget.sh
bash experiments/npe_mask_augmentation/npe_mask_augmentation_run_queue_high_sim_budget.sh
```

Each queue runs all problems, mechanisms, and fractions.

## Single Run

```bash
PYTHONPATH=src python experiments/npe_mask_augmentation/train.py \
  --config experiments/npe_mask_augmentation/low_sim_budget/glm/glm_npe_mask_augmentation_mcar_eps010_config.yaml
```

## Reference Metrics

Each config uses the 10 high-quality full-observation reference posteriors in
`references/reference_posteriors_v1`. The 10 reference observations are
deterministically masked with the dataset's missingness mechanism, transformed
into `[zero_imputed_x, mask]`, and compared against the full-data reference
posterior. These metrics are degradation-from-full-information diagnostics, not
exact missing-data posterior errors.

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
