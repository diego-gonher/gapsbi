# NPE Learned Imputation Experiment

Deterministic learned imputation + NPE baseline (Lueckmann-style / NPE-NN).

This baseline jointly trains:

- an imputer network that predicts missing x entries, and
- an NPE conditional density estimator on completed inputs.

It is missingness-mechanism-agnostic (no explicit MCAR/MAR/MNAR model).

## Core idea

For each batch:

1. Build imputer input: `[x_obs_zero_imputed_scaled, mask]`
2. Predict `x_hat_full_scaled`
3. Compose completed inputs:
   `x_completed = mask * x_obs_scaled + (1 - mask) * x_hat_full_scaled`
4. Optimize:
   `loss = npe_loss + lambda_recon * recon_loss`

Where `recon_loss` is MSE restricted to missing entries only.

## Imputer type

- `auto` selects:
  - `cnn` for time-series tasks such as `oup`, `lotka_volterra`, and legacy `ricker`
  - `mlp` for vector tasks such as `glm`, `glu`
- You can force `mlp` or `cnn` in config.

## Run

```bash
PYTHONPATH=src python experiments/npe_learned_imputation/train.py \
  --config experiments/npe_learned_imputation/oup/oup_npe_learned_imputation_mcar_eps010_config.yaml
```

## Outputs

Per seed:

- `training_summary.png`
- `posterior_samples.h5`
- `tarp.png`
- `sbc_rank_histograms.png`
- `diagnostics_arrays.npz`
- `summary.json`

Root:

- `all_results.json`
