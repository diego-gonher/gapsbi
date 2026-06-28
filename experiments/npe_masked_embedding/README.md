# Masked Embedding NPE

This experiment family contains two lightweight mask-aware NPE baselines:

- `npe_masked_pooling`: tokenizes `[x_zero_imputed_scaled, mask]`, adds learned feature embeddings, pools observed tokens with a masked mean, and appends a compact mask summary.
- `npe_masked_attention`: uses the same tokenization and mask-summary path, with one shallow self-attention layer before masked pooling.

Both methods keep the posterior estimator as NSF-NPE. They are intended as constrained, Simformer-inspired NPE baselines rather than full transformer-diffusion SBI methods.

Default embedding settings are intentionally small:

```yaml
token_dim: 32
context_dim: 64
mask_summary_dim: 16
num_layers: 1      # attention only
num_heads: 2       # attention only
```

Example:

```bash
PYTHONPATH=src python experiments/npe_masked_embedding/train.py \
  --config experiments/npe_masked_embedding/npe_masked_attention/full_sim_budget/oup/oup_npe_masked_attention_mcar_eps025_config.yaml
```
