# NPE versus NRE-C reference comparison

The analysis scripts consume saved reference-evaluation outputs and the existing NPE partial-reference aggregate CSVs. They do not rerun posterior sampling or reference metrics.

For the OUP/MCAR/10% campaign:

```bash
conda run -n gapsbi_env python scripts/aggregate_nre_c_comparison.py \
  --problem oup --mechanism mcar --fraction 0.10

env MPLCONFIGDIR=/tmp/gapsbi-matplotlib conda run -n gapsbi_env \
  python scripts/plot_npe_nre_c_comparison.py
```

Outputs default to `outputs/analysis/nre_c_comparison/oup_mcar_eps010/`: seed-level and aggregate CSVs for NPE, NRE-C, and the combined comparison; provenance metadata; and PDF/PNG figures under `figures/`. `--outputs-root`, `--npe-analysis-dir`, `--analysis-dir`, `--seeds`, `--budgets`, and `--reference-ids` select alternative saved campaigns. The plotting script accepts an aggregate CSV plus optional problem/mechanism/fraction labels, so its figure layout is not tied to OUP.

Within each seed, C2ST is averaged over reference observations and posterior mean shift and covariance-trace ratio use medians. Across seeds, reports use the arithmetic mean, sample standard deviation, and 95% CI `mean ± 1.96 × SD / sqrt(n)`. NPE aggregates are loaded from the established partial-reference analysis and checked against its seed-level CSV; NRE-C aggregates are calculated from the 10 saved reference rows and checked against each evaluator's saved aggregate.

The full-data baseline column includes only saved NPE/NRE-C full-data results that pass reference and seed checks. Those baseline metrics target `p(θ | x_full)` and use full-data references. Missing-data columns target `p(θ | x_obs, m)` and share the partial-data references. The figure labels this target difference and treats the baseline column as contextual rather than directly comparable to the missing-data columns.
