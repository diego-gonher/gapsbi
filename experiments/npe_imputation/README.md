# NPE Imputation Experiments

Naive imputation-only NPE baselines for missing-data benchmarks.

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

## Run

```bash
PYTHONPATH=src python experiments/npe_imputation/train.py \
  --config experiments/npe_imputation/glm_zero_config.yaml
```

## Outputs

Each seed directory contains:

- `training_summary.png`
- `posterior_samples.h5`
- `tarp.png`
- `sbc_rank_histograms.png`
- `diagnostics_arrays.npz`
- `summary.json`

Experiment root contains:

- `all_results.json`
