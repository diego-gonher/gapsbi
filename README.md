# GAPSBI: Gaps in Data for Simulation-Based Inference

GAPSBI is a benchmark and experimentation framework for simulation-based inference under missing data. It provides synthetic benchmark problems, MCAR/MAR/MNAR missingness mechanisms, reproducible HDF5 datasets, baseline NPE methods, calibration diagnostics, experiment aggregation, and posterior shift diagnostics.

The core dataset contract is simple:

- Generate complete simulator outputs `x_full`.
- Generate a binary mask where `1 = observed` and `0 = missing`.
- Store observed data as `x_obs = x_full * mask`.
- Save train/validation/test splits in a shared HDF5 format.
- Keep randomness explicit with reproducible seeds.

## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate gapsbi_env
```

Install the package in editable mode:

```bash
pip install -e .
```

The package requires Python `>=3.11`. Core dependencies include `numpy`, `h5py`, `matplotlib`, `scikit-learn`, `scipy`, `torch`, `sbi`, and `tqdm`.

## Current Benchmark Status

Implemented benchmark tasks:

- OUP
- GLM
- GLU
- Lotka-Volterra
- Ricker (legacy/implemented, not part of the current main four-problem benchmark)
- Spatial SIR
- Hodgkin-Huxley

Implemented missingness mechanisms:

- MCAR: pointwise and block masking
- MAR: coordinate/time-dependent masking
- MNAR: self-censoring value-dependent masking

Implemented methods:

- Full-data NPE
- Zero-imputation NPE
- Mean-imputation NPE
- Zero-imputation + mask augmentation NPE
- Masked-pooling embedding NPE
- Masked-attention embedding NPE
- Learned-imputation NPE
- GAPSBI-native RISE-style probabilistic imputation + NPE

Implemented evaluation and analysis:

- Per-seed posterior sampling
- TARP calibration diagnostics
- SBC rank diagnostics
- Benchmark result aggregation
- Full-data seed diagnostics
- MMD and C2ST posterior shift metrics
- Per-observation posterior shift metrics
- Dataset characterization diagnostics

## Benchmark V1

Benchmark V1 is the main missing-data SBI comparison suite.

Problems:

- OUP
- GLM
- GLU
- Lotka-Volterra

Missingness mechanisms:

- MCAR
- MAR
- MNAR

Missingness fractions:

- 10%
- 25%
- 50%

Methods:

- Full-data NPE
- Zero-imputation NPE
- Mean-imputation NPE
- Zero-imputation + mask augmentation NPE
- Masked-pooling embedding NPE
- Masked-attention embedding NPE
- Learned-imputation NPE
- GAPSBI-native RISE-style probabilistic imputation + NPE

The current experiment plan uses five fixed training seeds and three nested
simulation-budget regimes over the same canonical HDF5 datasets:

- `high_sim_budget`: 90k train / 10k validation / 1k test.
- `mid_sim_budget`: 9k train / 1k validation / full 1k test, using train/validation subsets.
- `low_sim_budget`: 900 train / 100 validation / full 1k test, using train/validation subsets.

Budget-specific runs should write to separate output roots such as
`outputs_high_sim_budget/`, `outputs_mid_sim_budget/`, and
`outputs_low_sim_budget/`.

## Implemented Simulators

### OUP

`OUPSimulator` implements a RISE-style Ornstein-Uhlenbeck process with a
slightly widened default equilibrium prior.

- `name`: `"oup"`
- `theta = [theta1, log_theta2]`
- `theta_dim = 2`
- default `x_shape = (25,)`
- prior:
  - `theta1 ~ Uniform(0, 2)`
  - `log_theta2 ~ Uniform(-2, 3)`

### GLU

`GLUSimulator` implements the Gaussian Linear Uniform benchmark.

- `name`: `"glu"`
- default `dim = 10`
- `theta_dim = dim`
- `x_shape = (dim,)`
- prior:
  - `theta_i ~ Uniform(-prior_bound, prior_bound)`
- simulator:
  - `x = theta + Normal(0, simulator_scale)`

### GLM

`GLMSimulator` implements a Bernoulli GLM with fixed stimulus features.

- `name`: `"glm"` for summary mode
- `name`: `"glm_raw"` for raw mode
- default `dim = 10`
- default `duration = 100`
- prior:
  - `theta[0] ~ Normal(0, 2)` using variance parameterization
  - `theta[1:] ~ Normal(0, inv(F.T @ F))`
  - `F[i, i] = 1 + sqrt(i / 9)`, `F[i, i - 1] = -2`, `F[i, i - 2] = 1` for the default 9 weights where those indices exist
- summary mode output:
  - `x_shape = (dim,)`
  - first entry is spike count
  - remaining entries are spike-triggered stimulus summaries
- raw mode output:
  - `x_shape = (duration,)`
  - binary spike train

### Lotka-Volterra

`LotkaVolterraSimulator` implements a two-population predator-prey ODE benchmark
with lognormal observation noise and a 1D packed observation vector.

- `name`: `"lotka_volterra"`
- `theta = [alpha, beta, gamma, delta]`
- `theta_dim = 4`
- default `num_timepoints = 50`
- default `days = 20`
- default `x_shape = (100,)`
- observation layout:
  - `x = [prey_t0, predator_t0, prey_t1, predator_t1, ..., prey_t49, predator_t49]`
- prior:
  - `log(theta) ~ Normal([-0.125, -3.0, -0.125, -3.0], 0.5^2 I)`
- observation model:
  - deterministic LV trajectory from initial state `[30, 1]`
  - lognormal observation noise with scale `0.1`

### Ricker

`RickerSimulator` implements a RISE-style stochastic Ricker population model. It
remains implemented for legacy experiments but is not part of the current main
four-problem benchmark.

- `name`: `"ricker"`
- `theta = [log_r, phi]`
- `theta_dim = 2`
- default `x_shape = (100,)`
- prior:
  - `log_r ~ Uniform(2, 8)`
  - `phi ~ Uniform(0, 20)`

### Spatial SIR

`SpatialSIRSimulator` implements a lightweight NumPy/SciPy spatial SIR lattice benchmark with one final snapshot observation.

- `name`: `"spatial_sir"`
- `theta = [beta, gamma]`
- `theta_dim = 2`
- default `lattice_shape = (16, 16)`
- default `measurement_time = 0.25`
- default `original_x_shape = (3, 16, 16)`
- default `x_shape = (768,)`
- prior:
  - `beta ~ Uniform(0, 1)`
  - `gamma ~ Uniform(0, 1)`
- observation:
  - one final susceptible/infected/recovered snapshot at `measurement_time`
  - channels are flattened for the HDF5 contract
  - it is not a full time series

For the default `16x16` lattice, `measurement_time=0.25` provides richer active infection structure than later times, which often let epidemics die out on the small grid.

### Hodgkin-Huxley

`HodgkinHuxleySimulator` implements a NumPy port of the Hodgkin-Huxley reference simulator with explicit RNG noise and downsampled raw voltage traces.

- `name`: `"hodgkin_huxley"`
- `theta = [g_Na, g_K]`
- `theta_dim = 2`
- prior:
  - `g_Na ~ Uniform(0.5, 80.0)`
  - `g_K ~ Uniform(1e-4, 15.0)`
- default raw simulation:
  - `duration = 120.0` ms
  - `dt = 0.01` ms
  - `t_on = 10.0` ms
  - `curr_level = 5e-4`
  - `downsample = 20`
- default `x_shape` is approximately `(601,)`
- output is a downsampled raw voltage trace in `float32`

Raw mode is the default because missingness over time is meaningful for voltage traces. Summary mode is not implemented.

## Missingness

Implemented MCAR masks:

- `PointMCARMask`: independent Bernoulli masking at each entry.
- `BlockMCARMask`: contiguous missing blocks for 1D vectors or batched 1D data.
- `LotkaVolterraTimeBlockMCARMask`: timestamp-block MCAR masking for LV. Missing timestamps hide both prey and predator entries.

Implemented MAR masks:

- `CoordinateMARMask`: coordinate/time-dependent missingness. It depends only on index metadata along the last axis, not on `x_full` values, so it is MAR rather than MNAR. Modes are `increasing`, `decreasing`, and `middle`. The requested `missing_fraction` is approximately the realized missing fraction before probability clipping; realized missingness can be lower when `max_probability` clips probabilities.
- `LotkaVolterraTimeMARMask`: timestamp-level LV MAR masking. A single time decision is expanded to both prey and predator entries.

Implemented MNAR masks:

- `SelfCensoringMNARMask`: value-dependent self-censoring. Each sample is optionally transformed, min/max shifted into `[0, 1]`, then higher normalized values receive higher missingness probability. Constant samples use score `0.5` everywhere to avoid division instability.
- `MeanNormalizedSelfCensoringMNARMask`: value-dependent self-censoring with the same min/max score, then per-sample mean normalization before applying the missingness fraction. This is the default generic MNAR mask for benchmark V1 because realized missingness is closer to the requested fraction.
- `LotkaVolterraLogTotalMNARMask`: timestamp-level LV MNAR masking based on `log(prey_t + predator_t)`, with the same mask applied to both populations at a timestamp.

`SelfCensoringMNARMask` supports `score_transform`:

- `identity`: score directly from `x_full`
- `log1p`: score from `np.log1p(x_full)`, useful for nonnegative spiky/count data such as legacy Ricker
- `abs`: score from `np.abs(x_full)`, useful for signed vector data when magnitude should drive missingness

Mask convention:

- `1`: observed
- `0`: missing

## HDF5 Dataset Contract

Saved datasets use this grouped layout:

```text
/train/theta
/train/x_full
/train/x_obs
/train/mask

/val/theta
/val/x_full
/val/x_obs
/val/mask

/test/theta
/test/x_full
/test/x_obs
/test/mask
```

Array conventions:

- `theta`: shape `(N, theta_dim)`
- `x_full`: shape `(N, *x_shape)`
- `x_obs`: same shape as `x_full`
- `mask`: same shape as `x_full`
- `x_obs = x_full * mask`

Metadata is stored as file-level HDF5 attributes. Nested simulator and mask metadata are stored as JSON strings and decoded on load.

## Canonical Dataset Configuration

The current benchmark V1 canonical dataset split is:

| Split | Size |
| --- | ---: |
| Train | 90,000 |
| Validation | 10,000 |
| Test | 1,000 |

Other constants:

- Dataset seed: `123`
- Missingness levels: `0.10`, `0.25`, `0.50`
- OUP/GLU/GLM masks:
  - MCAR: `point_mcar`
  - MAR: `coordinate_mar`, `mar-mode=increasing`
  - MNAR: mean-normalized `identity` self-censoring
- Lotka-Volterra masks:
  - MCAR: `lv_time_block_mcar`, `block-size=5`
  - MAR: `lv_time_mar`, `mar-mode=increasing`
  - MNAR: `lv_log_total_mnar`

Generate benchmark V1 canonical datasets with:

```bash
bash scripts/generate_benchmark_v1_datasets.sh
```

Mid- and low-budget experiments should reuse these HDF5 files and subset only the
train/validation splits in experiment configs. This keeps the full 1k test split
matched across budget ablations.

Toy examples below use smaller split sizes for quick local testing.

## Generate Datasets

Use `scripts/generate_dataset.py`.

Dataset generation shows per-simulation progress bars for each split by default using `tqdm`:

- `Generating train`
- `Generating val`
- `Generating test`

Use `--no-progress` to disable progress output, which is useful for tests and scripted runs.

Example Lotka-Volterra dataset with time-block MCAR:

```bash
PYTHONPATH=src python scripts/generate_dataset.py \
  --task lotka_volterra \
  --mask lv_time_block_mcar \
  --missing-fraction 0.25 \
  --block-size 5 \
  --n-train 1000 \
  --n-val 200 \
  --n-test 200 \
  --seed 123 \
  --output data/lotka_volterra_time_block_mcar_25.h5 \
  --overwrite
```

Example GLU dataset:

```bash
PYTHONPATH=src python scripts/generate_dataset.py \
  --task glu \
  --dim 10 \
  --simulator-scale 0.1 \
  --mask block_mcar \
  --block-size 3 \
  --missing-fraction 0.25 \
  --output data/glu_block_mcar_25.h5 \
  --overwrite
```

Example GLM dataset:

```bash
PYTHONPATH=src python scripts/generate_dataset.py \
  --task glm \
  --dim 10 \
  --duration 100 \
  --summary raw \
  --mask point_mcar \
  --missing-fraction 0.25 \
  --output data/glm_mcar_25.h5 \
  --overwrite
```

Example GLU dataset with value-dependent MNAR self-censoring:

```bash
PYTHONPATH=src python scripts/generate_dataset.py \
  --task glu \
  --dim 10 \
  --simulator-scale 0.1 \
  --mask self_censoring_mnar_mean_normalized \
  --missing-fraction 0.25 \
  --output data/glu_self_censoring_mean_normalized_mnar_25.h5 \
  --overwrite
```

Example Lotka-Volterra dataset with log-total MNAR:

```bash
PYTHONPATH=src python scripts/generate_dataset.py \
  --task lotka_volterra \
  --mask lv_log_total_mnar \
  --missing-fraction 0.25 \
  --output data/lotka_volterra_log_total_mnar_25.h5 \
  --overwrite
```

Example OUP dataset with coordinate-dependent MAR:

```bash
PYTHONPATH=src python scripts/generate_dataset.py \
  --task oup \
  --mask coordinate_mar \
  --missing-fraction 0.25 \
  --mar-mode increasing \
  --output data/oup_coordinate_mar_25.h5 \
  --overwrite
```

Example Spatial SIR dataset with coordinate-dependent MAR:

```bash
PYTHONPATH=src python scripts/generate_dataset.py \
  --task spatial_sir \
  --grid-size 16 \
  --mask coordinate_mar \
  --missing-fraction 0.25 \
  --output data/spatial_sir_coordinate_mar_25.h5 \
  --overwrite
```

Example Hodgkin-Huxley dataset with block MCAR:

```bash
PYTHONPATH=src python scripts/generate_dataset.py \
  --task hodgkin_huxley \
  --mask block_mcar \
  --missing-fraction 0.25 \
  --block-size 20 \
  --n-train 100 \
  --n-val 20 \
  --n-test 20 \
  --seed 123 \
  --output data/hodgkin_huxley_block_mcar_25.h5 \
  --overwrite
```

Supported generation options:

- `--task {ricker,oup,glu,glm,spatial_sir,hodgkin_huxley,lotka_volterra}`
- `--mask {point_mcar,block_mcar,self_censoring_mnar,self_censoring_mnar_mean_normalized,coordinate_mar,lv_time_block_mcar,lv_time_mar,lv_log_total_mnar}`
- `--missing-fraction`
- `--block-size`
- `--mar-mode {increasing,decreasing,middle}`
- `--mar-floor`
- `--mar-max-probability`
- `--mar-middle-width`
- `--mnar-score-transform {identity,log1p,abs}`
- `--n-train`
- `--n-val`
- `--n-test`
- `--seed`
- `--output`
- `--overwrite`
- `--progress`
- `--no-progress`

Task-specific options:

- GLU: `--dim`, `--simulator-scale`
- GLM: `--dim`, `--duration`, `--summary {sufficient,raw}`
- Lotka-Volterra: `--num-timepoints`, `--days`, `--observation-noise-scale`
- Spatial SIR: `--grid-size`, `--measurement-time`, `--simulation-step-size`, `--initial-infection-rate`
- Hodgkin-Huxley: `--duration`, `--dt`, `--t-on`, `--curr-level`, `--downsample`
- MAR masks: `--mar-mode`, `--mar-floor`, `--mar-max-probability`, `--mar-middle-width`
- MNAR masks: `--mnar-score-transform {identity,log1p,abs}`

For Spatial SIR, masking is applied at the spatial cell level. A single cell-level mask is expanded across the susceptible, infected, and recovered channels before flattening, so all three state channels for a cell are observed or missing together.

For Lotka-Volterra, masking is applied at the timestamp level. The 1D
observation vector is interleaved by population within each timestamp, and
LV-specific masks always observe or hide both prey and predator together.

## Dataset Diagnostics and Plotting

Plot example observations:

```bash
PYTHONPATH=src python scripts/plot_dataset_examples.py \
  --input data/lotka_volterra_time_block_mcar_25.h5 \
  --split train \
  --num-examples 6 \
  --seed 0 \
  --output outputs/lotka_volterra_examples.png
```

The plotting CLI supports `--plot-type auto`, `timeseries`, `vector`,
`spatial_sir`, and `lotka_volterra`. `auto` chooses spatial plots for Spatial
SIR, LV-specific two-population plots for Lotka-Volterra, vector plots for
GLU/GLM, and time-series plots otherwise.

Run dataset characterization diagnostics:

```bash
PYTHONPATH=src python scripts/diagnose_dataset.py \
  --dataset-path data/canonical_v1/oup/mcar/oup_mcar_eps025_seed123.h5 \
  --output-dir outputs/dataset_diagnostics/oup_mcar_eps025 \
  --split train
```

This produces summaries and plots for:

- prior/theta marginals
- missingness rates and missingness vs observation value
- `x_full` observation distributions
- theta-observation correlation heatmaps

Run lightweight mask-information diagnostics over generated benchmark datasets:

```bash
PYTHONPATH=src python scripts/diagnose_mask_information.py \
  data/canonical_v1/glu/mnar/glu_mnar_self_censoring_mean_normalized_identity_eps025_seed123.h5 \
  --sample-size 1000
```

For each dataset this writes `mask_information/` plots next to the dataset
generation diagnostics and updates
`outputs/dataset_generation_diagnostics/mask_information_summary.csv`.

## Baseline Experiments

All NPE experiments use predefined HDF5 train/validation/test splits, use `FixedSplitNPE_C` or the same fixed-split posterior construction, sample posteriors on held-out test data, and write per-seed diagnostics. The configured density estimator is `nsf` unless explicitly changed in a config.

Common per-seed outputs:

- `training_summary.png`
- `model_checkpoint.pt`
- `posterior_samples.h5`
- `tarp.png`
- `sbc_rank_histograms.png`
- `diagnostics_arrays.npz`
- `summary.json`

`model_checkpoint.pt` is a lightweight PyTorch checkpoint for the trained per-seed model. It stores the trained density-estimator weights, preprocessing scalers and scaling metadata, run config, dataset path, dimensions, and method-specific extras such as masked-embedding config or learned/RISE imputer weights. It is intended for reconstructing the trained model for later inference without rerunning training.

Theta preprocessing uses train-only `StandardScaler` for GLM and
Lotka-Volterra, and train-only `MinMaxScaler(feature_range=(-1, 1))` for
bounded-prior tasks. The scaled-space NPE prior is Gaussian with empirical train
covariance for GLM/Lotka-Volterra and `BoxUniform([-1, 1]^d)` for bounded-prior
tasks.

Experiment root output:

- `all_results.json`

### Full-data NPE

Uses `x_full` only and serves as the reference no-missingness baseline.
The configured full-data runs also sample posteriors on the ten fixed reference
observations for posterior-fidelity metrics.

```bash
PYTHONPATH=src python experiments/npe_full_data/train.py \
  --config experiments/npe_full_data/high_sim_budget/oup_config.yaml
```

Mid and low simulation budgets:

```bash
PYTHONPATH=src python experiments/npe_full_data/train.py \
  --config experiments/npe_full_data/mid_sim_budget/oup_config.yaml

PYTHONPATH=src python experiments/npe_full_data/train.py \
  --config experiments/npe_full_data/low_sim_budget/oup_config.yaml
```

### Zero/Mean Imputation NPE

Uses `x_obs` after scaling, then imputes missing entries in scaled x-space.

- `zero`: missing entries become `0.0`
- `mean`: missing entries become observed-only train feature means

```bash
PYTHONPATH=src python experiments/npe_imputation/train.py \
  --config experiments/npe_imputation/zero_imputation/full_sim_budget/oup/oup_zero_mcar_eps025_config.yaml
```

Low simulation budget:

```bash
PYTHONPATH=src python experiments/npe_imputation/train.py \
  --config experiments/npe_imputation/zero_imputation/low_sim_budget/oup/oup_zero_mcar_eps025_config.yaml
```

### Zero-imputation + Mask Augmentation

Uses zero-imputed scaled observations concatenated with the binary mask:

```text
x_aug = [x_imputed, mask]
```

```bash
PYTHONPATH=src python experiments/npe_mask_augmentation/train.py \
  --config experiments/npe_mask_augmentation/full_sim_budget/oup/oup_npe_mask_augmentation_mcar_eps025_config.yaml
```

Low simulation budget:

```bash
PYTHONPATH=src python experiments/npe_mask_augmentation/train.py \
  --config experiments/npe_mask_augmentation/low_sim_budget/oup/oup_npe_mask_augmentation_mcar_eps025_config.yaml
```

### Masked Embedding NPE

Uses the same zero-imputed scaled observation and binary mask condition as mask augmentation, but passes `[x_zero_imputed_scaled, mask]` through a small mask-aware embedding network before NSF-NPE. The pooling variant tokenizes each feature, adds learned feature embeddings, uses masked mean pooling over observed tokens, and appends a compact mask summary. The attention variant adds one shallow self-attention layer with missing entries excluded as attention keys/values, while still preserving the mask pattern through the mask-summary path.

The default configs intentionally keep the encoders small:

```text
token_dim = 32
context_dim = 64
mask_summary_dim = 16
masked_attention: num_layers = 1, num_heads = 2
```

```bash
PYTHONPATH=src python experiments/npe_masked_embedding/train.py \
  --config experiments/npe_masked_embedding/npe_masked_pooling/full_sim_budget/oup/oup_npe_masked_pooling_mcar_eps025_config.yaml
```

```bash
PYTHONPATH=src python experiments/npe_masked_embedding/train.py \
  --config experiments/npe_masked_embedding/npe_masked_attention/full_sim_budget/oup/oup_npe_masked_attention_mcar_eps025_config.yaml
```

Full-budget masked embedding outputs are written directly under `outputs/npe_masked_pooling/` and `outputs/npe_masked_attention/`. Low simulation-budget configs are under `experiments/npe_masked_embedding/{npe_masked_pooling,npe_masked_attention}/low_sim_budget/` and write under `outputs_low_sim_budget/npe_masked_pooling/` and `outputs_low_sim_budget/npe_masked_attention/`.

### Learned-imputation NPE

Trains an imputer network jointly with NPE. The imputer predicts missing x entries from `[x_obs_zero_imputed_scaled, mask]`, and the NPE model trains on completed inputs.

```bash
PYTHONPATH=src python experiments/npe_learned_imputation/train.py \
  --config experiments/npe_learned_imputation/full_sim_budget/oup/oup_npe_learned_imputation_mcar_eps025_config.yaml
```

Low simulation budget:

```bash
PYTHONPATH=src python experiments/npe_learned_imputation/train.py \
  --config experiments/npe_learned_imputation/low_sim_budget/oup/oup_npe_learned_imputation_mcar_eps025_config.yaml
```

### NPE RISE-style Probabilistic Imputation

Trains a lightweight probabilistic MLP imputer jointly with an NPE density estimator. The imputer consumes `[x_obs_scaled, mask]`, predicts a Gaussian completion distribution for `x`, and the NPE loss is optimized on completed inputs. An optional mask-prediction head adds a mask loss; `use_mask_head: auto` enables it for MNAR datasets and disables it for MCAR/MAR by default.

```bash
PYTHONPATH=src python experiments/npe_rise/train.py \
  --config experiments/npe_rise/full_sim_budget/oup/oup_npe_rise_mcar_eps025_config.yaml
```

Low simulation budget:

```bash
PYTHONPATH=src python experiments/npe_rise/train.py \
  --config experiments/npe_rise/low_sim_budget/oup/oup_npe_rise_mcar_eps025_config.yaml
```

## Evaluation Metrics

### Calibration

- **SBC**: simulation-based calibration ranks. Well-calibrated posteriors should produce approximately uniform rank histograms across posterior samples.
- **TARP**: tests posterior calibration through expected coverage probability curves. The repository stores TARP plots, TARP arrays, and aggregate errors such as MAE/IAE from the identity line.

### Posterior Shift

- **MMD**: maximum mean discrepancy between posterior sample distributions. The implemented scripts use an RBF kernel with the median heuristic. Larger MMD indicates larger distributional shift.
- **C2ST**: classifier two-sample test accuracy. A logistic classifier is trained to distinguish posterior samples from two methods or seeds. Accuracy near `0.5` indicates hard-to-distinguish posteriors; higher accuracy indicates stronger shift.
- **Euclidean posterior mean shift**: Euclidean distance between posterior means. This is a simple location-shift diagnostic.
- **Mahalanobis posterior mean shift**: posterior mean displacement measured in units of the matched full-data posterior covariance. Values near `0` indicate little location shift; values around `1` indicate roughly one posterior standard deviation of displacement.
- **Covariance trace ratio**: ratio of missing-data posterior covariance trace to full-data posterior covariance trace. Values above `1` indicate larger marginal posterior variance on average.
- **Log-determinant covariance ratio**: difference in log covariance determinants, `logdet(Sigma_missing) - logdet(Sigma_full)`. Positive values indicate posterior uncertainty-volume expansion; negative values indicate contraction.

Posterior shift is computed from existing `posterior_samples.h5` files and does not retrain models.

## Benchmark Analysis Scripts

Some analysis scripts still use legacy `campaign1` filenames. They are retained
for compatibility with existing experiment outputs.

Aggregate seed-level experiment outputs:

```bash
PYTHONPATH=src python scripts/aggregate_campaign1_results.py \
  --outputs-dir outputs \
  --out outputs/campaign1_master_results.csv
```

Aggregate low simulation budget outputs:

```bash
PYTHONPATH=src python scripts/aggregate_campaign1_results.py \
  --outputs-dir outputs_low_sim_budget \
  --out outputs_low_sim_budget/campaign1_master_results.csv
```

Launch full-budget and low-budget queue scripts with the matching budget-specific queues, for example:

```bash
bash experiments/npe_imputation/zero_imputation_oup_experiments_full_sim_budget.sh
bash experiments/npe_imputation/zero_imputation_oup_experiments_low_sim_budget.sh
```

Analyze stability and runtime across seeds/configurations:

```bash
PYTHONPATH=src python scripts/analyze_campaign1_stability.py \
  --input outputs/campaign1_master_results.csv \
  --outdir outputs/analysis
```

Report full-data baseline diagnostics:

```bash
PYTHONPATH=src python scripts/report_full_data_diagnostics.py \
  --input outputs/campaign1_master_results.csv \
  --out outputs/analysis/full_data_diagnostics_report.txt
```

Compute aggregate posterior shift between each missing-data run and the matched full-data run:

```bash
PYTHONPATH=src python scripts/compute_campaign1_shift_metrics.py \
  --input outputs/campaign1_master_results.csv \
  --out outputs/analysis/campaign1_shift_metrics.csv \
  --summary-out outputs/analysis/campaign1_shift_metrics_summary.csv
```

Compute full-data seed-to-seed posterior variability:

```bash
PYTHONPATH=src python scripts/compute_full_data_seed_shift_metrics.py \
  --input outputs/campaign1_master_results.csv \
  --out outputs/analysis/full_data_seed_shift_metrics.csv \
  --summary-out outputs/analysis/full_data_seed_shift_metrics_summary.csv
```

Compute per-observation posterior shift, comparing each test observation's missing-data posterior to the matched full-data posterior:

```bash
PYTHONPATH=src python scripts/compute_campaign1_per_observation_shift_metrics.py \
  --input outputs/campaign1_master_results.csv \
  --out outputs/analysis/campaign1_per_observation_shift_metrics.csv \
  --summary-out outputs/analysis/campaign1_per_observation_shift_metrics_summary.csv
```

Compute per-observation posterior moment shifts to separate posterior location changes from uncertainty-volume changes:

```bash
PYTHONPATH=src python scripts/compute_campaign1_per_observation_moment_shift_metrics.py \
  --input outputs/campaign1_master_results.csv \
  --out outputs/analysis/campaign1_per_observation_moment_shift_metrics.csv \
  --summary-out outputs/analysis/campaign1_per_observation_moment_shift_metrics_summary.csv
```

Compute the corresponding full-data per-observation seed variability baseline for posterior moment shifts:

```bash
PYTHONPATH=src python scripts/compute_full_data_per_observation_seed_moment_shift_metrics.py \
  --input outputs/campaign1_master_results.csv \
  --out outputs/analysis/full_data_per_observation_seed_moment_shift_metrics.csv \
  --summary-out outputs/analysis/full_data_per_observation_seed_moment_shift_metrics_summary.csv
```

## Python API Example

```python
from gapsbi.datasets import generate_dataset
from gapsbi.io import save_gapsbi_hdf5
from gapsbi.masks import LotkaVolterraTimeBlockMCARMask
from gapsbi.simulators import LotkaVolterraSimulator

simulator = LotkaVolterraSimulator()
mask = LotkaVolterraTimeBlockMCARMask(missing_fraction=0.25, block_size=5)

dataset = generate_dataset(
    simulator=simulator,
    mask_generator=mask,
    n_train=1000,
    n_val=200,
    n_test=200,
    seed=123,
)

save_gapsbi_hdf5(
    "data/lotka_volterra_time_block_mcar_25.h5",
    dataset,
    metadata={
        "task": simulator.name,
        "simulator": simulator.metadata(),
        "mask": mask.metadata(),
        "seed": 123,
    },
    overwrite=True,
)
```

## Tests

The test suite covers:

- RNG reproducibility
- prior sampling and log probabilities
- simulator shapes and reproducibility
- MCAR, MAR, and MNAR masks
- HDF5 contract validation and roundtrip behavior
- dataset generation CLI behavior
- dataset plotting helpers
- preprocessing/scaling helpers
- posterior sampling, SBC, and TARP helpers
- NPE helper wiring without slow training
- Benchmark aggregation and stability analysis
- posterior shift and moment-shift metric helpers and analysis scripts

Run tests with:

```bash
PYTHONPATH=src pytest
```

## Repository Layout

```text
configs/                         # Shared configs and scratch configuration
data/                            # Generated canonical and local datasets
experiments/
  npe_full_data/                 # Full-data NPE baseline
  npe_imputation/                # Zero/mean imputation baselines
  npe_mask_augmentation/         # Zero-imputation + mask baseline
  npe_masked_embedding/          # Masked pooling/attention embedding NPE baselines
  npe_learned_imputation/        # Learned-imputation baseline
  npe_rise/                      # GAPSBI-native RISE-style imputation + NPE
outputs_high_sim_budget/         # High simulation-budget outputs
outputs_mid_sim_budget/          # Mid simulation-budget outputs
outputs_low_sim_budget/          # Low simulation-budget outputs
outputs_local/                   # Local exploratory outputs
scripts/
  generate_dataset.py
  generate_benchmark_v1_datasets.sh
  plot_dataset_examples.py
  diagnose_dataset.py
  aggregate_campaign1_results.py
  analyze_campaign1_stability.py
  report_full_data_diagnostics.py
  compute_campaign1_shift_metrics.py
  compute_full_data_seed_shift_metrics.py
  compute_campaign1_per_observation_shift_metrics.py
  compute_campaign1_per_observation_moment_shift_metrics.py
  compute_full_data_per_observation_seed_moment_shift_metrics.py
src/gapsbi/
  diagnostics/                   # Dataset plotting helpers
  evaluation/                    # Posterior sampling, SBC, TARP
  masks/                         # MCAR/MAR/MNAR mask generators
  methods/                       # NPE and imputation utilities
  preprocessing/                 # Scaling helpers
  simulators/                    # Benchmark simulators
  utils/                         # Seeding utilities
tests/                           # Unit and integration tests
```

## Future Work

Planned or external extensions include:

- Simformer-style methods
- paper-complete or external RISE variants beyond the current GAPSBI-native baseline
- additional learned representation methods
- broader campaign coverage for Spatial SIR and Hodgkin-Huxley
- larger benchmark suites and collaborator-scale runs
