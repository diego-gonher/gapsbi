# Reference Posterior Artifacts

This document is the task-specific reference-generation companion to the
[canonical workflow guide](../docs/workflows.md). The versioned directories
below are current immutable provenance locations, not the intended long-term
public-facing names. No artifact paths are renamed in this documentation pass.

Reference posterior files are separate benchmark artifacts, not training datasets.
They store unscaled fixed observations and high-quality posterior samples for
reference-fidelity metrics such as C2ST, posterior mean shift, and covariance
scale comparisons.

The versioned path convention is:

```text
references/reference_posteriors_v1/{problem}_references.h5
```

Each HDF5 file uses this contract:

```text
/observations/theta_true                  (10, theta_dim), unscaled
/observations/x_full                      (10, x_dim), unscaled
/reference_posterior/theta_samples        (10, num_reference_samples, theta_dim), unscaled
/diagnostics/...                          task-specific sampler diagnostics
```

Required file attributes include `schema_name`, `schema_version`,
`reference_version`, `problem`, `posterior_method`, `num_observations`,
`num_reference_samples`, `theta_dim`, `x_dim`, `theta_scaled`, `x_scaled`,
`observation_seed`, `posterior_seed`, and simulator metadata.

Once a reference version is used for paper results, treat it as immutable. If the
simulator or posterior-generation procedure changes, create a new versioned
directory instead of overwriting old artifacts.

## Current Reference Problems

The current reference-posterior set is focused on the four main benchmark
problems: GLU, OUP, GLM, and Lotka-Volterra. Ricker remains implemented as a
legacy simulator but is not part of the current main reference set.

| Problem | Parameters | Prior | Default observation | Reference posterior |
| --- | --- | --- | --- | --- |
| GLU | `theta[0:10]` | independent `Uniform(-1, 1)` | `x = theta + Normal(0, 0.1)`, `x_dim=10` | analytic truncated Gaussian |
| OUP | `theta = [theta1, log_theta2]` | `theta1 ~ Uniform(0, 2)`, `log_theta2 ~ Uniform(-2, 3)` | RISE-style OUP path with widened equilibrium prior, `n=25`, `T=5`, `var=0.1`, `y0=10` | deterministic grid posterior |
| GLM | `theta = [beta, f[0:9]]` | `beta ~ Normal(0, 2)` using variance parameterization, `f ~ Normal(0, inv(F.T @ F))` with SBIBM Bernoulli-GLM `F` | raw Bernoulli spike train, `duration=100`, `stimulus_seed=42` | emcee MCMC |
| Lotka-Volterra | `theta = [alpha, beta, gamma, delta]` | `log(theta) ~ Normal([-0.125, -3.0, -0.125, -3.0], 0.5^2 I)` | interleaved prey/predator series, 50 timestamps over 20 days, lognormal observation noise scale `0.1` | emcee MCMC in log-parameter space |

All reference artifacts store unscaled `theta_true`, unscaled `x_full`, and
unscaled posterior samples. The HDF5 attributes also store simulator metadata,
including prior bounds, Gaussian prior parameters, or lognormal prior parameters.

Marginal diagnostic plots can be generated into the versioned artifact directory:

```bash
PYTHONPATH=src python scripts/plot_reference_posteriors.py \
  references/reference_posteriors_v1/glu_references.h5
```

By default this writes one PNG per reference observation under:

```text
references/reference_posteriors_v1/plots/
```

The plotting script writes marginal plots, corner/pair plots, stored MCMC trace
plots when available, and posterior predictive checks for GLU, OUP, GLM, and
Lotka-Volterra.

Write a compact diagnostics report with:

```bash
PYTHONPATH=src python scripts/report_reference_diagnostics.py \
  references/reference_posteriors_v1/glm_references.h5
```

The current OUP reference artifact uses a 4800 x 4800 final grid and 2400 x 2400
all-observation validation:

```bash
PYTHONPATH=src python scripts/generate_reference_posteriors.py \
  --problem oup \
  --grid-resolution 4800 \
  --validation-grid-resolution 2400 \
  --skip-spotcheck \
  --overwrite
```

The current GLM reference artifact uses raw GLM observations, 256 emcee walkers,
6000 burn-in steps, 6000 production steps, and two independent validation
ensembles for observations 0 and 1:

```bash
PYTHONPATH=src python scripts/generate_reference_posteriors.py \
  --problem glm \
  --num-walkers 256 \
  --burn-in-steps 6000 \
  --production-steps 6000 \
  --validation-num-ensembles 2 \
  --validation-observation-index 0 \
  --validation-observation-index 1 \
  --overwrite
```

The current Lotka-Volterra reference artifact uses emcee in log-parameter space,
64 walkers, 1500 burn-in steps, 4000 production steps, and two independent
validation ensembles for observations 0 and 1:

```bash
PYTHONPATH=src python scripts/generate_reference_posteriors.py \
  --problem lotka_volterra \
  --num-walkers 64 \
  --burn-in-steps 1500 \
  --production-steps 4000 \
  --validation-num-ensembles 2 \
  --validation-observation-index 0 \
  --validation-observation-index 1 \
  --overwrite
```

The current GLU reference artifact is analytic:

```bash
PYTHONPATH=src python scripts/generate_reference_posteriors.py \
  --problem glu \
  --overwrite
```

## GLM/MNAR independent full-reference replica

`glm_mnar_eps{010,025,050}_references.h5` implements the exact finite
compatible-completion posterior mixture. For 23 of its 30 reference entries,
the observed binary values and MNAR mask determine one compatible completion,
so the partial target equals the full-data target. The remaining seven entries
have an alternative completion with negligible estimated mass (at most
`1.48e-6`). This is a property of this specific self-censoring construction,
not a general claim about MNAR mechanisms.

The partial artifacts intentionally reuse canonical full-reference samples for
their dominant component. Therefore direct C2ST between those copied arrays is
invalid: identical rows can receive opposing labels in cross-validation folds.
The versioned independent replica
`glm_references_independent_replica_v1.h5` is a fresh, explicitly seeded
`emcee` run on the immutable canonical observations. It is used only by the
corrected direct reference-posterior analysis in
`outputs/analysis/reference_target_comparison/`; it does not alter estimator
benchmarks or any partial-data reference artifact.

The correction records duplicate-row preflight checks, old-full versus
independent-full null controls, and classifier-seed sensitivity. Independent
samples can produce small nonzero posterior mean shifts and non-unit trace
ratios even when the underlying targets coincide. At these sample sizes, C2ST
cannot resolve alternative mixture components with probabilities near `1e-6`.
