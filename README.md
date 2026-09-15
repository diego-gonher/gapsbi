# GapSBI: Gaps in Data in Simulation Based Inference

GapSBI is a benchmark and experimentation framework for simulation based
inference with missing observations. It includes synthetic benchmark tasks,
MCAR, MAR, and MNAR missingness mechanisms, reproducible HDF5 datasets, NPE
baselines, posterior diagnostics, result aggregation, and publication plotting
scripts.

## Installation

Create the conda environment and install the package in editable mode:

```bash
conda env create -f environment.yml
conda activate gapsbi_env
pip install -e .
```

GapSBI requires Python `>=3.11`. The main dependencies are declared in
`pyproject.toml` and `environment.yml`.

## Benchmark Scope

Core benchmark tasks:

* OUP
* GLM
* GLU
* Lotka Volterra

Extra simulators are retained for experimentation, but are not part of the
reported benchmark:

* Ricker
* Spatial SIR
* Hodgkin Huxley

Missingness mechanisms:

* MCAR: missing completely at random.
* MAR: coordinate or time dependent missingness.
* MNAR: value dependent self censoring.

Missing fractions:

* 10 percent
* 25 percent
* 50 percent

Core benchmark methods:

* Full data NPE
* Zero imputation NPE
* Learned constant imputation NPE
* Zero imputation plus mask augmentation NPE
* Probabilistic learned imputation NPE
* Masked transformer embedding NPE

Legacy or extra methods are kept for comparison and experimentation, but are not
part of the reported core benchmark:

* Mean imputation NPE
* Deterministic input dependent learned imputation NPE

Simulation budgets:

| Budget | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| `low_sim_budget` | 900 | 100 | 1,000 |
| `mid_sim_budget` | 9,000 | 1,000 | 1,000 |
| `high_sim_budget` | 90,000 | 10,000 | 1,000 |

All reported experiments use five training seeds:

```text
101, 202, 303, 404, 505
```

## Dataset Contract

GapSBI datasets are HDF5 files with train, validation, and test groups:

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

Convention:

* `mask = 1` means observed.
* `mask = 0` means missing.
* `x_obs = x_full * mask`.
* Metadata is stored as HDF5 attributes.

Generate the canonical benchmark datasets with:

```bash
bash scripts/generate_benchmark_v1_datasets.sh
```

Generate a custom dataset with:

```bash
PYTHONPATH=src python scripts/generate_dataset.py \
  --task oup \
  --mask coordinate_mar \
  --missing-fraction 0.25 \
  --mar-mode increasing \
  --output data/oup_coordinate_mar_25.h5 \
  --overwrite
```

Useful dataset generation options include:

* `--task {oup,glu,glm,lotka_volterra,ricker,spatial_sir,hodgkin_huxley}`
* `--mask {point_mcar,block_mcar,coordinate_mar,self_censoring_mnar_mean_normalized,lv_time_block_mcar,lv_time_mar,lv_log_total_mnar}`
* `--missing-fraction`
* `--n-train`, `--n-val`, `--n-test`
* `--seed`
* `--output`
* `--overwrite`

## Running Experiments

Each experiment family has YAML configs under `experiments/`. The examples below
run one OUP configuration. Use the run queue scripts in each experiment folder
for full grids.

Full data NPE:

```bash
PYTHONPATH=src python experiments/npe_full_data/train.py \
  --config experiments/npe_full_data/high_sim_budget/oup_config.yaml
```

Zero imputation NPE:

```bash
PYTHONPATH=src python experiments/npe_imputation/train.py \
  --config experiments/npe_imputation/zero_imputation/high_sim_budget/oup/oup_zero_mcar_eps025_config.yaml
```

Learned constant imputation NPE:

```bash
PYTHONPATH=src python experiments/npe_learned_constant_imputation/train.py \
  --config experiments/npe_learned_constant_imputation/high_sim_budget/oup/oup_npe_learned_constant_imputation_mcar_eps025_config.yaml
```

Mask augmentation NPE:

```bash
PYTHONPATH=src python experiments/npe_mask_augmentation/train.py \
  --config experiments/npe_mask_augmentation/high_sim_budget/oup/oup_npe_mask_augmentation_mcar_eps025_config.yaml
```

Probabilistic learned imputation NPE:

```bash
PYTHONPATH=src python experiments/npe_probabilistic_learned_imputation/train.py \
  --config experiments/npe_probabilistic_learned_imputation/high_sim_budget/oup/oup_npe_probabilistic_learned_imputation_mcar_eps025_config.yaml
```

Masked transformer embedding NPE:

```bash
PYTHONPATH=src python experiments/npe_masked_transformer_embedding/train.py \
  --config experiments/npe_masked_transformer_embedding/high_sim_budget/oup/oup_npe_masked_transformer_embedding_mcar_eps025_config.yaml
```

Typical per seed outputs include:

* `summary.json`
* `model_checkpoint.pt`
* `posterior_samples.h5`
* `reference_posterior_samples.h5` when reference evaluation is enabled
* `diagnostics_arrays.npz`
* diagnostic plots

## Evaluation and Analysis

The benchmark uses two metric families.

Reference comparison metrics compare inferred posteriors with complete data
reference posteriors:

* C2ST
* posterior mean shift
* covariance trace ratio

Calibration metrics use held out simulations with known parameters:

* TARP empirical coverage curves
* TARP MAE and IAE
* SBC rank diagnostics

Compile seed level summaries:

```bash
PYTHONPATH=src python scripts/compile_experiment_results.py
```

Add calibration diagnostics from `diagnostics_arrays.npz`:

```bash
PYTHONPATH=src python scripts/add_diagnostic_metrics.py
```

Aggregate results across seeds:

```bash
PYTHONPATH=src python scripts/aggregate_experiment_results.py
```

The main analysis files are written under:

```text
outputs/analysis/
outputs/analysis/aggregates/
```

## Plotting

Generate scalar benchmark metric figures:

```bash
PYTHONPATH=src python scripts/plot_benchmark_summary.py --all
```

Generate combined MCAR and MNAR benchmark figures:

```bash
PYTHONPATH=src python scripts/plot_benchmark_summary.py --combined-mcar-mnar --all
```

Generate TARP coverage curves:

```bash
PYTHONPATH=src python scripts/plot_coverage_curves.py --all
```

Generate computational cost figures:

```bash
PYTHONPATH=src python scripts/plot_computational_cost.py --all
```

Generate posterior example figures:

```bash
PYTHONPATH=src python scripts/plot_posterior_examples.py --problem oup
PYTHONPATH=src python scripts/plot_posterior_examples.py --problem glm
PYTHONPATH=src python scripts/plot_posterior_examples.py --problem glu
PYTHONPATH=src python scripts/plot_posterior_examples.py --problem lv
```

Figures are written under:

```text
outputs/analysis/figures/
```

## Diagnostics

Plot example observations:

```bash
PYTHONPATH=src python scripts/plot_dataset_examples.py \
  --input data/canonical_v1/oup/mcar/oup_mcar_eps025_seed123.h5 \
  --split train \
  --num-examples 6 \
  --output outputs/oup_examples.png
```

Run dataset diagnostics:

```bash
PYTHONPATH=src python scripts/diagnose_dataset.py \
  --dataset-path data/canonical_v1/oup/mcar/oup_mcar_eps025_seed123.h5 \
  --output-dir outputs/dataset_diagnostics/oup_mcar_eps025 \
  --split train
```

Run mask information diagnostics:

```bash
PYTHONPATH=src python scripts/diagnose_mask_information.py \
  data/canonical_v1/glu/mnar/glu_mnar_self_censoring_mean_normalized_identity_eps025_seed123.h5 \
  --sample-size 1000
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

Run the test suite with:

```bash
PYTHONPATH=src pytest
```

The tests cover simulators, masks, dataset IO, preprocessing, posterior sampling,
evaluation helpers, experiment setup, aggregation scripts, and plotting scripts.

## Repository Layout

```text
experiments/                    Experiment configs and training entry points
references/                     Reference posterior documentation
scripts/                        Dataset, analysis, diagnostic, and plotting scripts
src/gapsbi/                     Package source
tests/                          Unit and integration tests
environment.yml                 Conda environment
pyproject.toml                  Package metadata
CAMPAIGNS.md                    Experiment grid notes
```

Generated datasets, outputs, reference artifacts, and paper figures are not
tracked by default.
