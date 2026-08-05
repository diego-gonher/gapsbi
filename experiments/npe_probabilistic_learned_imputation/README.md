# NPE Probabilistic Learned Imputation

RISE-inspired probabilistic learned-imputation baseline for benchmark v1.

This experiment trains an MLP latent neural-process-style imputer jointly with
the shared NSF-NPE backbone. The imputer consumes `[x_obs_scaled, mask]`, samples
a latent `z`, predicts a Gaussian completion model for `x_full_scaled`, and
passes the mean-completed observation to NPE:

```text
x_completed = mask * x_obs_scaled + (1 - mask) * mean(x_full_scaled | x_obs_scaled, mask, z)
```

For MNAR datasets, `use_mask_head: auto` enables an additional mask-prediction
loss. MCAR and MAR use the same architecture; MAR differs only through the
dataset mask generator.

The imputer is intentionally MLP-only to keep method capacity comparable across
baselines. This is therefore RISE-inspired, not a faithful reproduction of the
official RISE architecture.

## Generate Configs

```bash
PYTHONPATH=src python experiments/npe_probabilistic_learned_imputation/generate_configs.py
```

## Run Queues

```bash
bash experiments/npe_probabilistic_learned_imputation/npe_probabilistic_learned_imputation_run_queue_low_sim_budget.sh
bash experiments/npe_probabilistic_learned_imputation/npe_probabilistic_learned_imputation_run_queue_mid_sim_budget.sh
bash experiments/npe_probabilistic_learned_imputation/npe_probabilistic_learned_imputation_run_queue_high_sim_budget.sh
```

## Outputs

Each seed saves posterior samples, model checkpoint, TARP/SBC diagnostics, and
per-reference C2ST, posterior mean shift, and covariance trace ratio against the
ten fixed reference posteriors.
