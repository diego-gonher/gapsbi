# Experimental OUP NRE-C learned-constant imputation

This experiment uses the same per-feature scaled-space constants as
`npe_learned_constant_imputation`, initialized at zero and jointly optimized
with SBI NRE-C's standard contrastive loss. The mask selects observed values;
it is not concatenated to the classifier input.

```bash
conda run -n gapsbi_env python experiments/nre_c_learned_constant_imputation/train.py \
  --config experiments/nre_c_learned_constant_imputation/low_sim_budget/oup_mcar_eps010_config.yaml
```

The configured smoke test uses IDs 0 and 1. Run `evaluate.py` with
`--reference-indices 0 1 2 3 4 5 6 7 8 9` to evaluate all references without
retraining. Outputs are ignored experimental artifacts.
