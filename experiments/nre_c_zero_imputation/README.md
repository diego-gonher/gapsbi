# Experimental OUP NRE-C zero imputation

This isolated experiment trains NRE-C on the canonical OUP MCAR-10% split after
the same **scaled-space zero imputation** used by `npe_zero_imputation`. The
25-dimensional classifier condition is only the imputed observation; it never
receives the mask. Evaluation targets the partial-data posterior
`p(theta | x_obs, m)` using the matching partial-reference artifact.

Run the two-reference smoke test:

```bash
conda run -n gapsbi_env python experiments/nre_c_zero_imputation/train.py \
  --config experiments/nre_c_zero_imputation/low_sim_budget/oup_mcar_eps010_config.yaml
```

Evaluate an existing checkpoint (all canonical IDs, without retraining):

```bash
conda run -n gapsbi_env python experiments/nre_c_zero_imputation/evaluate.py \
  --config experiments/nre_c_zero_imputation/low_sim_budget/oup_mcar_eps010_config.yaml \
  --checkpoint outputs_low_sim_budget/nre_c_zero_imputation/oup/mcar/oup_mcar_eps010/seed_101/model_checkpoint.pt \
  --reference-indices 0 1 2 3 4 5 6 7 8 9
```

Outputs are deliberately ignored experimental artifacts. The grid uses 400 and
800 resolutions for diagnostics and samples the selected 800² posterior with
within-cell jitter.
