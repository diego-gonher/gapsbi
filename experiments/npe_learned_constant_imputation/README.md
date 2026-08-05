# NPE Learned-Constant Imputation Experiment

Lueckmann-style learned constant imputation baseline for benchmark v1
missing-data tasks.

This experiment learns one scalar constant per observation feature in scaled
x-space. Missing entries are replaced by those learned constants, and the NPE
estimator is trained on the completed observations without receiving the mask as
an explicit conditioning input.

For mask convention `1 = observed`, `0 = missing`, the completed input is:

```text
x_completed = mask * x_obs_scaled + (1 - mask) * c
```

where `c` is a trainable vector of length `x_dim`, optimized jointly with the NPE
negative log likelihood. There is no reconstruction network and no auxiliary
reconstruction loss.

## Experiment Grid

Active configs cover:

- problems: OUP, GLM, GLU, Lotka-Volterra
- budgets: low `900/100`, mid `9000/1000`, high full `90000/10000`
- missingness: MCAR, MAR, MNAR
- missing fractions: `0.10`, `0.25`, `0.50`
- seeds: `101, 202, 303, 404, 505`

Configs are generated from:

```bash
PYTHONPATH=src python experiments/npe_learned_constant_imputation/generate_configs.py
```

## Run Queues

```bash
bash experiments/npe_learned_constant_imputation/npe_learned_constant_imputation_run_queue_low_sim_budget.sh
bash experiments/npe_learned_constant_imputation/npe_learned_constant_imputation_run_queue_mid_sim_budget.sh
bash experiments/npe_learned_constant_imputation/npe_learned_constant_imputation_run_queue_high_sim_budget.sh
```

## Single Run

```bash
PYTHONPATH=src python experiments/npe_learned_constant_imputation/train.py \
  --config experiments/npe_learned_constant_imputation/low_sim_budget/glm/glm_npe_learned_constant_imputation_mcar_eps010_config.yaml
```

## Reference Metrics

Each config uses the 10 high-quality full-observation reference posteriors in
`references/reference_posteriors_v1`. The 10 reference observations are
deterministically masked with the dataset's missingness mechanism, completed with
the learned constants, and compared against the full-data reference posterior.
These metrics are degradation-from-full-information diagnostics, not exact
missing-data posterior errors.

Saved per-reference metrics:

- C2ST accuracy
- posterior mean shift
- covariance trace ratio

Each seed also stores `learned_constants_scaled` in posterior/reference HDF5
outputs and in the model checkpoint extras.
