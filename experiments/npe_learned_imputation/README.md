# NPE Learned Imputation Experiment

Deterministic input-dependent learned imputation + NPE baseline.

Public-repo note: this is a legacy/extra experiment and is not part of the final
GapSBI benchmark or paper results. It is retained for experimentation and
historical comparison. Some configs use older task/budget conventions, including
`full_sim_budget` and the legacy `ricker` task, so they should not be treated as
drop-in members of the final four-task benchmark grid.

This is not the Lueckmann-style learned constant baseline. For that method, use
`experiments/npe_learned_constant_imputation/`.

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

These commands are examples for the legacy baseline. They are not required for
reproducing the final benchmark figures.

```bash
PYTHONPATH=src python experiments/npe_learned_imputation/train.py \
  --config experiments/npe_learned_imputation/low_sim_budget/oup/oup_npe_learned_imputation_mcar_eps010_config.yaml
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
