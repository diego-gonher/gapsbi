# Experiment Plan

Problems:

* OUP
* GLM
* GLU
* LV

Missingness:

* MCAR
* MAR
* MNAR

Fractions:

* 10%
* 25%
* 50%

Five Training Seeds:
* 101
* 202
* 303
* 404
* 505

1. `npe_full_data`: Full-data NPE (baseline; DONE)
2. `npe_zero_imputation`: Zero Imputation NPE (Wang et al. 2024; DONE)
3. `npe_learned_constant_imputation`: Per-Feautre Learned Imputation with NPE (Lueckman et al. 2017; low and mid budgets done)
4. `npe_mask_augmentation`: Zero-imputation + mask augmentation NPE (Wang et al. 2024)
5. `npe_probabilistic_learned_imputation`: Probabilistic Learned Imputation NPE (RISE-inspired)
6. `npe_masked_transformer_embedding`: Transfomer Embeddings with NPE

Here is another way of conceptualizing these methods:

| Family | Method | Core idea |
|--------|--------|-----------|
| **Oracle** | Full-data NPE | Upper bound with no missing data. |
| **Naive deterministic imputation** | Zero Imputation | Fill missing values with a constant. |
| | Per-feature Mean Imputation | Fill with empirical feature means. |
| **Learned deterministic imputation** | Learned Per-feature Imputation | Learn fixed replacement values jointly with NPE. |
| **Imputation + missingness indicators** | Zero Imputation + Mask Augmentation | Keep naive imputation but explicitly tell the network what was missing. |
| **Probabilistic imputation** | Probabilistic Learned Imputation | Learn \(p(x_{\rm miss}\mid x_{\rm obs})\) with a RISE-inspired latent imputer. |
| **Representation learning** | Transformer Embedding + NPE | Never reconstruct missing values; directly learn a representation robust to masks. |

Simulation budgets:

* `high_sim_budget`: 90k train / 10k validation / 1k test, with outputs under `outputs_high_sim_budget/` unless a config explicitly uses another full-budget root.
* `mid_sim_budget`: 9k train / 1k validation / 1k test, with outputs under `outputs_mid_sim_budget/` unless a config explicitly uses another full-budget root.
* `low_sim_budget`: 0.9k train / 0.1k validation / full 1k test, with outputs under `outputs_low_sim_budget/`.

Metrics to use:
* C2ST on the 10 high quality reference posteriors. This is a standard metric.
* Mean shift with the 10 reference posteriors. This is to measure any biases. 
* Covariance trace ratio with the 10 reference posteriors. This is to measure posterior broadening.
* TARP on a set of 1000 synthetic observations. This is to measure global posterior calibration.

Results workflow:
* Compile seed-level summaries with `scripts/compile_experiment_results.py`.
* Add diagnostics-array metrics with `scripts/add_diagnostic_metrics.py`.
* Aggregate seed-level results with `scripts/aggregate_experiment_results.py`.
* Generate publication figures with:
  * `scripts/plot_benchmark_summary.py` for scalar benchmark metrics.
  * `scripts/plot_coverage_curves.py` for mean TARP coverage curves.
  * `scripts/plot_computational_cost.py` for training and inference cost.
  * `scripts/plot_posterior_examples.py` for qualitative posterior examples.
