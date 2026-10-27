# Experimental full-data NRE-A

This is a narrow OUP/NRE-A feasibility baseline, not a validated benchmark
method. It trains SBI 0.26.1 `NRE_A` on the canonical low-budget 900/100
full-data split. Training automatically performs an OUP-only grid posterior
evaluation against the full-data references; it does not perform coverage or
missing-data training.

```bash
conda run -n gapsbi_env env PYTHONPATH=src python \
  experiments/nre_a_full_data/train.py \
  --config experiments/nre_a_full_data/low_sim_budget/oup_config.yaml
```

Evaluate an existing checkpoint without retraining:

```bash
conda run -n gapsbi_env env PYTHONPATH=src python \
  experiments/nre_a_full_data/evaluate.py \
  --config experiments/nre_a_full_data/low_sim_budget/oup_config.yaml \
  --checkpoint outputs_low_sim_budget/nre_a_full_data/oup_full_data/seed_101/model_checkpoint.pt
```

The evaluator records 400, 800, and 1600 midpoint-grid moment diagnostics,
uses the configured converged resolution for 10,000 within-cell-jittered draws
per reference, and writes `reference_posterior_samples.h5` and
`reference_evaluation.json` next to the ignored checkpoint.
