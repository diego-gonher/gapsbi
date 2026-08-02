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

Methods:

1. Full-data NPE
2. Mean-imputation NPE
3. Mean-imputation + mask augmentation NPE
4. Learned Imputation with NPE
5. RISE based NPE
6. Transfomer Embeddings with NPE

Simulation budgets:

* `high_sim_budget`: 90k train / 10k validation / 1k test, with outputs under `outputs_high_sim_budget/` unless a config explicitly uses another full-budget root.
* `mid_sim_budget`: 9k train / 1k validation / 1k test, with outputs under `outputs_mid_sim_budget/` unless a config explicitly uses another full-budget root.
* `low_sim_budget`: 0.9k train / 0.1k validation / full 1k test, with outputs under `outputs_low_sim_budget/`.

Metrics to use:
* C2ST on the 10 high quality reference posteriors. This is a standard metric.
* Mean shift with the 10 reference posteriors. This is to measure any biases. 
* Covariance trace ratio with the 10 reference posteriors. This is to measure posterior broadening.
* TARP on a set of 1000 synthetic observations. This is to measure global posterior calibration.
