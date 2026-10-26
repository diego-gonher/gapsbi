# GapSBI Benchmark and Reproduction Guide

GapSBI measures posterior inference when observations are incomplete. The
release benchmark asks whether an estimator approximates the posterior for the
actual retained observation and its mask, not the posterior available before
missingness.

## Scientific scope

The validated release contains GLU, OUP, Bernoulli GLM, and Lotka--Volterra at
10%, 25%, and 50% nominal missingness under MCAR and MNAR. The six validated
methods are Full-data NPE (a full-information comparator), zero imputation,
learned-constant imputation, mask augmentation, probabilistic learned
imputation, and masked-transformer embedding. Mean imputation and
deterministic learned imputation remain implemented research baselines; they
are not release benchmark methods because they lack complete partial-reference
campaigns.

For complete data, the reference target is

```text
p(theta | x_full) ∝ p(theta) p(x_full | theta).
```

For the release missing-data benchmark, it is

```text
p(theta | x_obs, m) ∝ p(theta) ∫ p(x_full | theta) p(m | x_full, theta) dx_mis,
```

where `x_obs = x_full * m`, `m=1` is observed, and the integral is over
missing entries. MCAR has mask probability independent of values; MNAR retains
the mask likelihood. GLM/MNAR therefore uses the informative self-censoring
mask and finite compatible-completion posterior mixture.

| Task | Full-data reference | Partial-data approach |
| --- | --- | --- |
| GLU | Analytic truncated Gaussian | Analytic MCAR; importance-sampled latent completion for MNAR. |
| OUP | Deterministic grid posterior | Grid conditioning for MCAR; importance-weighted missing trajectories for MNAR. |
| GLM | `emcee` MCMC | Observed-data MH for MCAR; exact finite completion mixture for MNAR. |
| Lotka--Volterra | `emcee` in log-parameter space | Screened observed-data MCMC for MCAR; parameter-level importance reweighting for MNAR. |

At LV/MNAR 50%, only original reference IDs `[0,1,3,4,5,6,8,9]` have
feasible partial references. This is missing-reference availability, not an
estimator failure, and corresponding aggregates use eight observations.

## Reproduction

```bash
conda env create -f environment.yml
conda run -n gapsbi_env pip install -e .
conda run -n gapsbi_env env PYTHONPATH=src python -m pytest tests -q

conda run -n gapsbi_env bash scripts/generate_benchmark_v1_datasets.sh
conda run -n gapsbi_env env PYTHONPATH=src python scripts/generate_reference_posteriors.py --problem glu
conda run -n gapsbi_env env PYTHONPATH=src python \
  scripts/generate_glu_mcar_partial_data_reference_posteriors.py --missing-fraction 0.25

# Select a committed method configuration.
conda run -n gapsbi_env env PYTHONPATH=src python \
  experiments/npe_imputation/train.py --config \
  experiments/npe_imputation/zero_imputation/low_sim_budget/glu/glu_zero_mcar_eps025_config.yaml

conda run -n gapsbi_env env PYTHONPATH=src python scripts/evaluate_partial_references.py \
  --problem glu --mechanism mcar --partial-only
conda run -n gapsbi_env env PYTHONPATH=src python scripts/analyze_partial_reference_comparison.py \
  --problem glu --mechanism mcar
conda run -n gapsbi_env python scripts/report_evaluation_failures.py
conda run -n gapsbi_env env PYTHONPATH=src python scripts/plot_benchmark_summary.py --all
```

The exact reference-generation matrix is in the
[partial-reference artifact README](../references/partial_data_references_v1/README.md).
Versioned release artifacts are immutable. MCMC and importance procedures are
reproducible from tracked source, seeds, and metadata, but not guaranteed
bitwise identical across operating systems, BLAS libraries, or parallel MCMC
scheduling.

## Evaluation, failures, and direct target analysis

`scripts/evaluate_partial_references.py --partial-only` is authoritative. It
checks persisted unscaled/scaled samples, checkpoint and dataset provenance,
observation/mask identity, and sparse original IDs. C2ST is the finite-row
mean; posterior mean shift and covariance-trace ratio are finite-row medians.
Nonfinite estimator draws produce invalid rows and reduce finite counts; scores
are never imputed.

`scripts/report_evaluation_failures.py` writes
`outputs/analysis/evaluation_reliability/evaluation_failure_summary.csv` plus
seed-level details. It distinguishes unavailable reference posteriors from
nonfinite estimator evaluations, so LV/MNAR 50% is not misreported as a
numerical failure.

For the diagnostic direct comparison, run:

```bash
conda run -n gapsbi_env env PYTHONPATH=src python \
  scripts/analyze_full_vs_partial_references.py
```

It writes `outputs/analysis/reference_target_comparison/` without replacing
historical analyses. GLM/MNAR uses an independent full-reference MCMC replica
because its partial artifact reuses dominant canonical full draws. The analysis
retains all 238 matched comparisons, including the 208 unaffected rows, and
records bootstrap intervals, null controls, duplicate checks, and seed
sensitivity.

## Historical and experimental components

| Component | Status and boundary |
| --- | --- |
| MAR masks, datasets, and configurations | Retained research extension; excluded from validated partial-reference benchmark plots and tables. |
| L-C2ST pilots | Experimental feasibility/sensitivity diagnostic; not a primary metric. |
| LV/MNAR joint-MCMC, SMC, and importance pilots | Exploratory sampler diagnostics; not reference generators or benchmark estimators. |
| `evaluate_glu_mcar_partial_references.py` | Historical checkpoint-resampling payload reproducer; saved-draw evaluator is authoritative. |
| Mean and deterministic learned imputation | Implemented extra baselines, not validated release methods. |
| Extra simulators | Retained for research; outside the four-task release scope. |

These components are not invoked by the commands above and must not be
presented as release benchmark results. Historical training metrics against
`p(theta | x_full)` remain degradation diagnostics, not final benchmark
metrics.
