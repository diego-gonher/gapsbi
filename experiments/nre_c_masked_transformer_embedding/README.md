# Experimental OUP NRE-C masked-transformer embedding

Uses the exact shared `MaskedTransformerEmbedding` used by the NPE method with
SBI `classifier_nn(..., embedding_net_x=..., z_score_x="none")`. The encoder
and NRE-C ResNet are one SBI ratio-estimator state dict and train jointly.

```bash
conda run -n gapsbi_env python experiments/nre_c_masked_transformer_embedding/evaluate.py \
  --config experiments/nre_c_masked_transformer_embedding/low_sim_budget/oup_mcar_eps010_config.yaml \
  --checkpoint outputs_low_sim_budget/nre_c_masked_transformer_embedding/oup/mcar/oup_mcar_eps010/seed_101/model_checkpoint.pt
```
