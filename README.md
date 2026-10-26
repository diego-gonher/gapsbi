# GapSBI: Gaps in Data in Simulation Based Inference

GapSBI is a benchmark for simulation-based inference with missing
observations. The validated release benchmark covers OUP, GLM, GLU, and
Lotka--Volterra under MCAR and MNAR missingness at 10%, 25%, and 50% nominal
missingness. Its primary posterior-fidelity evaluation uses partial-data
references conditioned on the realized observation and mask.

The repository also retains MAR configurations, exploratory simulators, and
diagnostic work. Those components are useful research history, but are outside
the validated release benchmark. See [the workflow guide](docs/workflows.md)
for the authoritative workflow and scope boundaries.

## Installation

Create the conda environment and install the package in editable mode:

```bash
conda env create -f environment.yml
conda activate gapsbi_env
pip install -e .
```

GapSBI requires Python `>=3.11`. The main dependencies are declared in
`pyproject.toml` and `environment.yml`.

## Validated Benchmark Scope

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

Missingness mechanisms in the release benchmark:

* MCAR: missing completely at random.
* MNAR: value dependent self censoring.

Missing fractions:

* 10 percent
* 25 percent
* 50 percent

Validated method inventory:

* Full-data NPE, retained as the full-information comparator.
* Zero-imputation NPE.
* Learned-constant-imputation NPE.
* Zero-imputation plus mask-augmentation NPE.
* Probabilistic learned-imputation NPE.
* Masked-transformer-embedding NPE.

The following implementations and configuration families are retained, but are
not asserted to be part of the validated partial-reference result inventory:

* Mean imputation NPE
* Deterministic input dependent learned imputation NPE

They are not excluded merely because they are older or non-default: the current
validated output trees and partial-reference analyses contain the six methods
listed above. Releasing either retained method as a benchmark method requires a
complete MCAR/MNAR partial-reference evaluation campaign.

MAR remains implemented and reproducible as a limitation/extension, not a
release benchmark mechanism. It has no validated partial-reference benchmark
claim in this release.

Simulation budgets:

| Budget | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| `low_sim_budget` | 900 | 100 | 1,000 |
| `mid_sim_budget` | 9,000 | 1,000 | 1,000 |
| `high_sim_budget` | 90,000 | 10,000 | 1,000 |

Validated missing-data campaigns use five training seeds:

```text
101, 202, 303, 404, 505
```

## Canonical Workflow

The authoritative current entry points are listed below. Existing paths retain
their versioned artifact names for provenance; this pass does not rename or move
scientific artifacts. The intended unversioned public names and migration
dependencies are documented in [docs/workflows.md](docs/workflows.md).

1. Generate datasets with `scripts/generate_benchmark_v1_datasets.sh`.
2. Generate full-data references with `scripts/generate_reference_posteriors.py`.
3. Generate partial-data references with the task/mechanism-specific
   `scripts/generate_*_partial_data_reference_posteriors.py` entry points.
4. Train with the appropriate `experiments/*/train.py` and committed YAML
   configuration.
5. Evaluate saved estimator draws against partial references with
   `scripts/evaluate_partial_references.py --partial-only`.
6. Aggregate partial-reference comparisons with
   `scripts/analyze_partial_reference_comparison.py` and create benchmark plots
   with `scripts/plot_benchmark_summary.py --reference-mode partial`.
7. Run `scripts/analyze_full_vs_partial_references.py` for the separate
   reference-target diagnostic; it applies the authoritative independent
   full-reference correction for GLM/MNAR.

Training-time reference metrics compare masked estimator inputs with
full-data references. They are preserved as historical full-information
diagnostics, not the final partial-data benchmark target.

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

Generate the currently canonical dataset collection with:

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

## Training

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

The final missing-data benchmark compares saved estimator posterior draws with
partial-data references for the same realized `(x_obs, mask)` pair. Its three
primary metrics are C2ST, posterior mean shift, and covariance-trace ratio.
The established aggregation convention is mean C2ST and median mean-shift and
trace-ratio across finite reference observations, followed by summaries across
training seeds.

Historical training-time full-reference metrics remain in existing summaries;
they must not be substituted for the partial-data result.

The repository also uses calibration diagnostics:

* TARP empirical coverage curves, MAE, and IAE.
* SBC rank diagnostics.

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

For the final partial-reference workflow, use the command sequence in
[docs/workflows.md](docs/workflows.md). It validates estimator/reference
provenance before computing metrics and preserves the sparse LV/MNAR reference
mapping.

The main analysis files are written under:

```text
outputs/analysis/
outputs/analysis/aggregates/
```

## Plotting

Generate the release-facing partial-reference figures (the default, MCAR/MNAR
only):

```bash
conda run -n gapsbi_env env PYTHONPATH=src python scripts/plot_benchmark_summary.py --all
```

Generate historical full-reference scalar figures explicitly:

```bash
conda run -n gapsbi_env env PYTHONPATH=src python scripts/plot_benchmark_summary.py \
  --reference-mode full --all --mechanisms mcar mar mnar
```

Generate combined MCAR and MNAR release figures with partial-data references:

```bash
conda run -n gapsbi_env env PYTHONPATH=src python scripts/plot_benchmark_summary.py \
  --combined-mcar-mnar --all
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

## Diagnostics and Limitations

The full-versus-partial-reference study is diagnostic only. Run
`scripts/analyze_full_vs_partial_references.py`; it uses the independent
full-reference replica for GLM/MNAR because the partial artifact can
intentionally reuse canonical full-reference draws. It writes a new,
public-facing `outputs/analysis/reference_target_comparison/` directory and
does not overwrite the versioned historical analyses.

At LV/MNAR 50%, only original reference indices `[0, 1, 3, 4, 5, 6, 8, 9]`
have feasible partial references. Aggregates report the available eight
observations; this is not interchangeable with a ten-observation comparison.
Estimator sampling failures are represented as non-finite draws and invalid
reference-metric rows; reporting retains finite counts and failure provenance
rather than imputing scores. MAR and L-C2ST are outside the primary benchmark.

See [docs/workflows.md](docs/workflows.md) for diagnostic and historical
boundaries.

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
docs/                           Canonical workflow and release-scope documentation
scripts/                        Dataset, analysis, diagnostic, and plotting scripts
src/gapsbi/                     Package source
tests/                          Unit and integration tests
environment.yml                 Conda environment
pyproject.toml                  Package metadata
CAMPAIGNS.md                    Experiment grid notes
```

Generated datasets, outputs, reference artifacts, and paper figures are not
tracked by default.
