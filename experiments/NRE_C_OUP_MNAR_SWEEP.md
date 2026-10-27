# OUP NRE-C MNAR 10% campaign

The MNAR matrix uses the canonical mean-normalized self-censoring dataset and the matching OUP partial-reference posterior artifact. It schedules the five NRE-C missing-data methods across three budgets and five seeds, for 75 sequential configurations.

Inspect the schedule without materializing files or running experiments:

```bash
experiments/nre_c_oup_mnar_sweep.sh --dry-run
```

When approved for execution, launch the resumable campaign with:

```bash
experiments/nre_c_oup_mnar_sweep.sh
```

Progress, failure records, generated per-job configurations, and the campaign lock are isolated under `logs/nre_c_oup_mnar_10pct/`. Check progress with `experiments/nre_c_oup_mnar_sweep.sh --progress` and resume using the same launch command.

The probabilistic learned-imputation override matches the NPE MNAR configuration: `lambda_np=1`, `lambda_mask=1`, and the informative-mask head is enabled. Checkpoints, output trees, and references are all MNAR-specific; no MCAR checkpoint can satisfy the campaign provenance checks.
