# Experimental full-data OUP NRE-C

NRE-C counterpart to the low-budget full-data OUP NRE-A feasibility baseline.
It uses SBI's default ResNet and the persisted 900/100 split. NRE-C defaults
are made explicit here: `num_classes=5`, `gamma=1.0`, yielding the standard
multi-class-sigmoid objective. Outputs are experimental and ignored.

```bash
conda run -n gapsbi_env env PYTHONPATH=src python experiments/nre_c_full_data/train.py \
  --config experiments/nre_c_full_data/low_sim_budget/oup_config.yaml
```
