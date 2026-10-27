# Experimental OUP NRE-C mask augmentation

This experiment applies the canonical NPE mask-augmentation representation:
`[zero_imputed_scaled_x_obs, binary_mask]`. It targets the OUP MCAR-10%
partial-data posterior, `p(theta | x_obs, m)`.

```bash
conda run -n gapsbi_env python experiments/nre_c_mask_augmentation/train.py \
  --config experiments/nre_c_mask_augmentation/low_sim_budget/oup_mcar_eps010_config.yaml
```

To evaluate an existing checkpoint for all references, append
`--reference-indices 0 1 2 3 4 5 6 7 8 9` to `evaluate.py`. Experimental
outputs are ignored and use 400²/800² grid diagnostics with an 800² posterior.
