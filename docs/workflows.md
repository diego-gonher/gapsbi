# Canonical Benchmark Workflows

## Scope and method inventory

The validated GapSBI benchmark comprises OUP, GLM, GLU, and Lotka--Volterra
under MCAR and MNAR at nominal missingness fractions 10%, 25%, and 50%.
The final missing-data target is the posterior
`p(theta | x_obs, mask)` for the realized partial observation and mask.

The verified benchmark-result inventory is:

| Role | Method identifier |
| --- | --- |
| Full-information comparator | `npe_full_data` |
| Missing-data benchmark | `npe_zero_imputation` |
| Missing-data benchmark | `npe_learned_constant_imputation` |
| Missing-data benchmark | `npe_mask_augmentation` |
| Missing-data benchmark | `npe_probabilistic_learned_imputation` |
| Missing-data benchmark | `npe_masked_transformer_embedding` |

`npe_mean_imputation` and `npe_learned_imputation` remain implemented and have
configuration families. They are not omitted because of their names or age;
they are not listed above because the current validated output inventory and
partial-reference analyses do not contain a complete MCAR/MNAR campaign for
them. They may be added only with a comparable validated campaign.

MAR code and configurations remain available for research use, but MAR is
outside the validated benchmark scope. L-C2ST is an experimental diagnostic,
not a primary benchmark metric.

## Authoritative workflow

| Stage | Current authoritative entry point | Result |
| --- | --- | --- |
| Dataset generation | `scripts/generate_benchmark_v1_datasets.sh` | Canonical HDF5 datasets. |
| Full-data references | `scripts/generate_reference_posteriors.py` | `p(theta | x_full)` reference artifacts. |
| Partial-data references | `scripts/generate_*_partial_data_reference_posteriors.py` | Task/mechanism-specific `p(theta | x_obs, mask)` artifacts. |
| Training | `experiments/*/train.py` with committed YAML | Checkpoint, posterior draws, calibration diagnostics, historical full-reference diagnostics. |
| Final estimator evaluation | `scripts/evaluate_partial_references.py --partial-only` | Provenance-checked partial-reference metric rows and aggregates. |
| Partial-reference comparison | `scripts/analyze_partial_reference_comparison.py` | Seed-level and aggregate comparisons, failure reporting, and diagnostic figures. |
| Benchmark plots | `scripts/plot_benchmark_summary.py --reference-mode partial` | Partial-reference benchmark figures without rewriting historical metrics. |
| Direct reference-target study | `scripts/analyze_full_vs_partial_references.py`, then `scripts/analyze_glm_mnar_independent_full_vs_partial.py` | Diagnostic full-versus-partial study; the corrected GLM/MNAR analysis is authoritative. |

The first four stages have task-specific details by design. In particular, the
reference samplers differ between analytic, grid, MCMC, importance, and
finite-completion targets. This documentation pass does not consolidate those
algorithms.

## Evaluation targets and metrics

There are two distinct reference targets:

* **Historical full-data diagnostic:** during training, an estimator receives a
  masked or completed reference input and is compared with `p(theta | x_full)`.
  Existing values are retained for provenance and degradation-from-full-
  information analyses.
* **Final partial-data benchmark:** post-hoc evaluation compares saved
  estimator draws with `p(theta | x_obs, mask)` for the exact persisted
  observation, mask, parameter units, checkpoint, and reference indices.

The final benchmark metrics are C2ST accuracy, posterior mean shift, and
covariance-trace ratio. Existing validated aggregation is preserved: C2ST is
averaged across finite reference observations; posterior mean shift and
covariance-trace ratio use their median. Seed-level values are then summarized
across training seeds. No metric definition or numerical result is changed by
this documentation.

The post-hoc evaluator is authoritative because it validates the saved
unscaled/scaled estimator draws, checkpoint provenance, full-reference
observation identity, partial-reference conditioning, and sparse original
reference indices before computing metrics.

## Known limitations and failure handling

* LV/MNAR at 50% has only eight feasible partial references: original indices
  `[0, 1, 3, 4, 5, 6, 8, 9]`. Its partial aggregate reports those eight entries;
  it is not a ten-reference matched comparison.
* Estimator sampling can fail or produce non-finite values. The evaluation
  records invalid rows and finite counts; it does not fabricate replacement
  scores.
* The GLM/MNAR partial artifact intentionally reuses canonical full-reference
  draws for dominant completion components. Direct C2ST between copied rows is
  invalid. The independent-replica generation and corrected GLM/MNAR analysis
  are therefore mandatory for that diagnostic study.
* MAR is not a validated release mechanism. L-C2ST, LV sampler pilots, and
  related investigations are diagnostic research components, not benchmark
  metrics.

## Intended public naming

The current physical paths such as `reference_posteriors_v1`,
`partial_data_references_v1`, and `full_vs_partial_references_v2` are immutable
provenance locations. They remain unchanged in this pass.

The path helpers in `gapsbi.references` centralize versioned defaults and allow
an explicit root for public aliases.  Public interfaces should use these
unversioned names when aliases are released:

| Concept | Intended public name |
| --- | --- |
| Full-data references | `references/full/` |
| Partial-data references | `references/partial/` |
| Final estimator metrics | `outputs/benchmark/partial_reference_metrics/` |
| Benchmark figures | `outputs/benchmark/figures/` |
| Direct reference-target diagnostics | `outputs/analysis/reference_target_comparison/` |
| Research diagnostics | `outputs/diagnostics/` |

Artifact release identifiers, HDF5 schema versions, seeds, hashes, and sampler
settings remain in provenance metadata. Existing ignored artifacts must not be
renamed in place.

## Reproduction boundary

The repository intentionally regenerates HDF5 reference artifacts from code
rather than distributing posterior samples. The partial-reference README
contains the complete eight-command MCAR/MNAR regeneration matrix. Use stored
artifact metadata and committed source together when reproducing a validated
campaign.

## Historical and experimental components

Keep, but do not treat as release entry points:

* MAR masks, datasets, and configurations.
* Mean and deterministic learned-imputation experiment families until they have
  a complete partial-reference campaign.
* LV/MNAR sampler pilots, validation scripts, and targeted follow-ups.
* L-C2ST feasibility and sensitivity pilots.
* Extra simulators outside OUP, GLM, GLU, and Lotka--Volterra.
