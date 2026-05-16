# NPE Mask-Augmentation Experiment

Wang-style mask augmentation baseline for missing-data NPE with `sbi`.

This experiment uses:

1. `x_obs` scaled with scaler fitted on `x_full_train`
2. zero imputation in scaled space
3. mask concatenation:
   `x_aug = [x_imputed, mask]`

Method recorded in outputs:

- `npe_mask_augmentation`

## Canonical preprocessing

1. Load `theta`, `x_full`, `x_obs`, `mask` for train/val/test.
2. Scale `theta` with train-only `MinMaxScaler(-1, 1)`.
3. Fit x scaler on `x_full_train` only:
   - GLM/GLU/OUP: `StandardScaler`
   - Ricker: `log1p + StandardScaler`
4. Transform `x_obs` using that fitted scaler.
5. Zero-impute missing entries in scaled space.
6. Concatenate `[x_imputed, mask.float()]` along feature axis.
7. Train `FixedSplitNPE_C` on augmented inputs.

## Run

```bash
PYTHONPATH=src python experiments/npe_mask_augmentation/train.py \
  --config experiments/npe_mask_augmentation/glm_config.yaml
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
