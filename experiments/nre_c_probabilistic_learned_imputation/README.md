# Experimental OUP NRE-C probabilistic learned imputation

Uses the canonical RISE MLP (including its latent draw and auxiliary losses) with
SBI NRE-C. Training resamples the latent imputation per batch. Each grid posterior
uses one fixed, seed-derived imputation draw per reference ID so its ratio is a
well-defined reproducible grid function; it does not integrate over imputer draws.

```bash
conda run -n gapsbi_env python experiments/nre_c_probabilistic_learned_imputation/train.py --config experiments/nre_c_probabilistic_learned_imputation/low_sim_budget/oup_mcar_eps010_config.yaml
```
