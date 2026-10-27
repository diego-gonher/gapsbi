# Experimental full-data NRE-A

This is a narrow OUP/NRE-A feasibility baseline, not a validated benchmark
method. It trains SBI 0.26.1 `NRE_A` on the canonical low-budget 900/100
full-data split and saves only the ratio estimator and training provenance.
It deliberately performs no posterior grid evaluation, reference comparison,
coverage study, or missing-data training.

```bash
conda run -n gapsbi_env env PYTHONPATH=src python \
  experiments/nre_a_full_data/train.py \
  --config experiments/nre_a_full_data/low_sim_budget/oup_config.yaml
```
