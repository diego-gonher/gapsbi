# Masked Transformer Embedding NPE

This experiment family contains a lightweight mask-aware transformer embedding
baseline. It tokenizes `[x_zero_imputed_scaled, mask]`, adds learned feature
embeddings, applies a shallow transformer encoder with missing entries excluded
as attention keys/values, then applies learned attention pooling over observed
tokens before NSF-NPE.

The posterior estimator is kept fixed as NSF-NPE. This is intended as a
constrained, transformer-embedding baseline rather than a full Simformer-style
inference method.

Canonical preprocessing matches the other NPE baselines:

- `theta` uses train-only `StandardScaler` for GLM/Lotka-Volterra and train-only `MinMaxScaler(feature_range=(-1, 1))` for bounded-prior tasks.
- `x` uses train-only `StandardScaler` for GLM/GLU/OUP/Lotka-Volterra; legacy Ricker uses `log1p + StandardScaler`.
- The scaled-space NPE prior is Gaussian with empirical train covariance for GLM/Lotka-Volterra and `BoxUniform([-1, 1]^d)` for bounded-prior tasks.

Default embedding settings are intentionally small:

```yaml
token_dim: 32
context_dim: 64
num_layers: 1
num_heads: 2
ff_multiplier: 2
```

Example:

```bash
PYTHONPATH=src python experiments/npe_masked_transformer_embedding/train.py \
  --config experiments/npe_masked_transformer_embedding/low_sim_budget/oup/oup_npe_masked_transformer_embedding_mcar_eps025_config.yaml
```

Budget queues:

```bash
bash experiments/npe_masked_transformer_embedding/npe_masked_transformer_embedding_run_queue_low_sim_budget.sh
bash experiments/npe_masked_transformer_embedding/npe_masked_transformer_embedding_run_queue_mid_sim_budget.sh
bash experiments/npe_masked_transformer_embedding/npe_masked_transformer_embedding_run_queue_high_sim_budget.sh
```
