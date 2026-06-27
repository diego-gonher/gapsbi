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
- Ricker
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
- Learned-imputation NPE
- GAPSBI-native RISE-style probabilistic imputation + NPE

Implemented evaluation and analysis:

- Per-seed posterior sampling
- TARP calibration diagnostics
- SBC rank diagnostics
- Campaign 1 result aggregation
- Full-data seed diagnostics
- MMD and C2ST posterior shift metrics
- Per-observation posterior shift metrics
- Dataset characterization diagnostics

## Campaign 1

Campaign 1 is the main completed baseline campaign for cheap, reproducible missing-data SBI comparisons.

Problems:

- OUP
- GLM
- GLU
- Ricker

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
- Learned-imputation NPE
- GAPSBI-native RISE-style probabilistic imputation + NPE

Each configuration uses 10 fixed random seeds. Campaign 1 is organized into simulation-budget-specific config trees:

- `full_sim_budget`: the canonical 45k train / 5k validation / 1k test setup.
- `low_sim_budget`: a 4.5k train / 500 validation ablation using the same HDF5 datasets and full test split.

Full-budget runs write under `outputs/` unless a config explicitly uses another full-budget root. Low-budget runs write under `outputs_low_sim_budget/` so ablations do not mix with full-budget results.

## Implemented Simulators

### Ricker

`RickerSimulator` implements a RISE-style stochastic Ricker population model.

- `name`: `"ricker"`
- `theta = [log_r, phi]`
- `theta_dim = 2`
- default `x_shape = (100,)`
- prior:
  - `log_r ~ Uniform(2, 8)`
  - `phi ~ Uniform(0, 20)`

### OUP

`OUPSimulator` implements a RISE-style Ornstein-Uhlenbeck process.

- `name`: `"oup"`
- `theta = [theta1, log_theta2]`
- `theta_dim = 2`
- default `x_shape = (25,)`
- prior:
  - `theta1 ~ Uniform(0, 2)`
  - `log_theta2 ~ Uniform(-2, 2)`

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
  - `theta_i ~ Uniform(-prior_bound, prior_bound)`
- summary mode output:
  - `x_shape = (dim,)`
  - first entry is spike count
  - remaining entries are spike-triggered stimulus summaries
- raw mode output:
  - `x_shape = (duration,)`
  - binary spike train

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

Implemented MAR masks:

- `CoordinateMARMask`: coordinate/time-dependent missingness. It depends only on index metadata along the last axis, not on `x_full` values, so it is MAR rather than MNAR. Modes are `increasing`, `decreasing`, and `middle`. The requested `missing_fraction` is approximately the realized missing fraction before probability clipping; realized missingness can be lower when `max_probability` clips probabilities.

Implemented MNAR masks:

- `SelfCensoringMNARMask`: value-dependent self-censoring. Each sample is optionally transformed, min/max shifted into `[0, 1]`, then higher normalized values receive higher missingness probability. Constant samples use score `0.5` everywhere to avoid division instability.

`SelfCensoringMNARMask` supports `score_transform`:

- `identity`: score directly from `x_full`
- `log1p`: score from `np.log1p(x_full)`, useful for nonnegative spiky/count data such as Ricker
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

The current Campaign 1 canonical dataset split is:

| Split | Size |
| --- | ---: |
| Train | 45,000 |
| Validation | 5,000 |
| Test | 1,000 |

Other constants:

- Dataset seed: `123`
- Missingness levels: `0.10`, `0.25`, `0.50`
- MCAR: `point_mcar`
- MAR: `coordinate_mar`, `mar-mode=increasing`
- MNAR:
  - `identity` self-censoring for GLU, GLM, OUP
  - `log1p` self-censoring for Ricker

Generate Campaign 1 canonical datasets with:

```bash
bash scripts/generate_campaign1_45k_datasets.sh
```

The repository also contains older/reference generation commands for larger `90k / 10k / 1k` datasets. Treat those as archival or extended-size runs unless you intentionally want the larger split.

Toy examples below use smaller split sizes for quick local testing.

## Generate Datasets

Use `scripts/generate_dataset.py`.

Dataset generation shows per-simulation progress bars for each split by default using `tqdm`:

- `Generating train`
- `Generating val`
- `Generating test`

Use `--no-progress` to disable progress output, which is useful for tests and scripted runs.

Example Ricker dataset with point MCAR:

```bash
PYTHONPATH=src python scripts/generate_dataset.py \
  --task ricker \
  --mask point_mcar \
  --missing-fraction 0.25 \
  --n-train 1000 \
  --n-val 200 \
  --n-test 200 \
  --seed 123 \
  --output data/ricker_mcar_25.h5 \
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
  --prior-bound 2.0 \
  --duration 100 \
  --summary sufficient \
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
  --mask self_censoring_mnar \
  --missing-fraction 0.25 \
  --output data/glu_self_censoring_mnar_25.h5 \
  --overwrite
```

Example Ricker dataset with log-compressed MNAR self-censoring:

```bash
PYTHONPATH=src python scripts/generate_dataset.py \
  --task ricker \
  --mask self_censoring_mnar \
  --missing-fraction 1.0 \
  --mnar-score-transform log1p \
  --output data/ricker_self_censoring_mnar_log1p.h5 \
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

- `--task {ricker,oup,glu,glm,spatial_sir,hodgkin_huxley}`
- `--mask {point_mcar,block_mcar,self_censoring_mnar,coordinate_mar}`
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
- GLM: `--dim`, `--prior-bound`, `--duration`, `--summary {sufficient,raw}`
- Spatial SIR: `--grid-size`, `--measurement-time`, `--simulation-step-size`, `--initial-infection-rate`
- Hodgkin-Huxley: `--duration`, `--dt`, `--t-on`, `--curr-level`, `--downsample`
- MAR masks: `--mar-mode`, `--mar-floor`, `--mar-max-probability`, `--mar-middle-width`
- MNAR masks: `--mnar-score-transform {identity,log1p,abs}`

For Spatial SIR, masking is applied at the spatial cell level. A single cell-level mask is expanded across the susceptible, infected, and recovered channels before flattening, so all three state channels for a cell are observed or missing together.

## Dataset Diagnostics and Plotting

Plot example observations:

```bash
PYTHONPATH=src python scripts/plot_dataset_examples.py \
  --input data/ricker_mcar_25.h5 \
  --split train \
  --num-examples 6 \
  --seed 0 \
  --output outputs/ricker_examples.png
```

The plotting CLI supports `--plot-type auto`, `timeseries`, `vector`, and `spatial_sir`. `auto` chooses spatial plots for Spatial SIR, vector plots for GLU/GLM, and time-series plots otherwise.

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

## Baseline Experiments

All NPE experiments use predefined HDF5 train/validation/test splits, use `FixedSplitNPE_C` or the same fixed-split posterior construction, sample posteriors on held-out test data, and write per-seed diagnostics. The configured density estimator is `nsf` unless explicitly changed in a config.

Common per-seed outputs:

- `training_summary.png`
- `posterior_samples.h5`
- `tarp.png`
- `sbc_rank_histograms.png`
- `diagnostics_arrays.npz`
- `summary.json`

Experiment root output:

- `all_results.json`

### Full-data NPE

Uses `x_full` only and serves as the reference no-missingness baseline.

```bash
PYTHONPATH=src python experiments/npe_full_data/train.py \
  --config experiments/npe_full_data/full_sim_budget/oup_config.yaml
```

Low simulation budget:

```bash
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

### RISE-style Probabilistic Imputation + NPE

Trains a lightweight probabilistic MLP imputer jointly with an NPE density estimator. The imputer consumes `[x_obs_scaled, mask]`, predicts a Gaussian completion distribution for `x`, and the NPE loss is optimized on completed inputs. An optional mask-prediction head adds a mask loss; `use_mask_head: auto` enables it for MNAR datasets and disables it for MCAR/MAR by default.

```bash
PYTHONPATH=src python experiments/rise/train.py \
  --config experiments/rise/full_sim_budget/oup/oup_rise_mcar_eps025_config.yaml
```

Low simulation budget:

```bash
PYTHONPATH=src python experiments/rise/train.py \
  --config experiments/rise/low_sim_budget/oup/oup_rise_mcar_eps025_config.yaml
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

## Campaign 1 Analysis Scripts

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
from gapsbi.masks import PointMCARMask
from gapsbi.simulators import RickerSimulator

simulator = RickerSimulator()
mask = PointMCARMask(missing_fraction=0.25)

dataset = generate_dataset(
    simulator=simulator,
    mask_generator=mask,
    n_train=1000,
    n_val=200,
    n_test=200,
    seed=123,
)

save_gapsbi_hdf5(
    "data/ricker_mcar_25.h5",
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
- Campaign 1 aggregation and stability analysis
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
  npe_learned_imputation/        # Learned-imputation baseline
  rise/                          # GAPSBI-native RISE-style imputation + NPE
outputs/                         # Campaign outputs and analysis products
outputs_low_sim_budget/          # Low simulation-budget campaign outputs
outputs_local/                   # Local exploratory outputs
scripts/
  generate_dataset.py
  generate_campaign1_45k_datasets.sh
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
