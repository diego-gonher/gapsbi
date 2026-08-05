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

1. Full-data NPE (baseline; DONE)
2. Zero Imputation NPE (Wang et al. 2024)
2. Per-Feature Mean Imputation NPE (left to appendix; DONE)
3. Zero-imputation + mask augmentation NPE (Wang et al. 2024)
4. Per-Feautre Learned Imputation with NPE (Lueckman et al. 2017)
5. RISE based NPE
6. Transfomer Embeddings with NPE

Here is another way of conceptualizing these methods:

| Family | Method | Core idea |
|--------|--------|-----------|
| **Oracle** | Full-data NPE | Upper bound with no missing data. |
| **Naive deterministic imputation** | Zero Imputation | Fill missing values with a constant. |
| | Per-feature Mean Imputation | Fill with empirical feature means. |
| **Learned deterministic imputation** | Learned Per-feature Imputation | Learn fixed replacement values jointly with NPE. |
| **Imputation + missingness indicators** | Zero Imputation + Mask Augmentation | Keep naive imputation but explicitly tell the network what was missing. |
| **Probabilistic imputation** | RISE | Learn \(p(x_{\rm miss}\mid x_{\rm obs})\) and marginalize posterior uncertainty. |
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
