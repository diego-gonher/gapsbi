# GAPSBI: Gaps in Data for Simulation-Based Inference

GAPSBI is a small benchmark-generation toolkit for simulation-based inference with missing data. The current repository focuses on producing reproducible synthetic datasets with explicit masks, saving them in a common HDF5 format, and providing quick inspection plots.

The core design is:

- Generate complete simulator outputs `x_full`.
- Generate a binary mask where `1 = observed` and `0 = missing`.
- Store observed data as `x_obs = x_full * mask`.
- Keep randomness explicit with `numpy.random.Generator` objects.

## Current Status

Implemented:

- Simulator interface: `gapsbi.simulators.base.Simulator`
- Simulators:
  - `RickerSimulator`
  - `OUPSimulator`
  - `GLUSimulator`
  - `GLMSimulator`
- Priors:
  - `UniformPrior`
  - `LogUniformPrior`
- RNG utilities:
  - `make_rng`
  - `split_rng`
  - `make_rngs`
- Mask generators:
  - `PointMCARMask`
  - `BlockMCARMask`
- Dataset generation:
  - `generate_split`
  - `generate_dataset`
- HDF5 I/O:
  - `save_gapsbi_hdf5`
  - `load_gapsbi_hdf5`
  - `validate_gapsbi_dataset`
- Diagnostics and plotting:
  - time-series example plots
  - vector example plots
  - dataset plotting CLI
- CLIs:
  - `scripts/generate_dataset.py`
  - `scripts/plot_dataset_examples.py`

Not implemented yet:

- MAR and MNAR masks
- Spatial SIR and Weinberg simulators
- registry helpers
- baselines
- calibration diagnostics such as SBC/TARP

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

The package currently requires Python `>=3.11`.

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

## Missingness

Implemented MCAR masks:

- `PointMCARMask`: independent Bernoulli masking at each entry.
- `BlockMCARMask`: contiguous missing blocks for 1D vectors or batched 1D data.

Mask convention:

- `1`: observed
- `0`: missing

MAR and MNAR generators are planned but not implemented yet.

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

## Generate Datasets

Use `scripts/generate_dataset.py`.

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

Supported generation options:

- `--task {ricker,oup,glu,glm}`
- `--mask {point_mcar,block_mcar}`
- `--missing-fraction`
- `--block-size`
- `--n-train`
- `--n-val`
- `--n-test`
- `--seed`
- `--output`
- `--overwrite`

Task-specific options:

- GLU: `--dim`, `--simulator-scale`
- GLM: `--dim`, `--prior-bound`, `--duration`, `--summary {sufficient,raw}`

## Plot Datasets

Use `scripts/plot_dataset_examples.py`.

Example:

```bash
PYTHONPATH=src python scripts/plot_dataset_examples.py \
  --input data/ricker_mcar_25.h5 \
  --split train \
  --num-examples 6 \
  --seed 0 \
  --output outputs/ricker_examples.png
```

The plotting CLI supports:

- `--plot-type auto`
- `--plot-type timeseries`
- `--plot-type vector`

`auto` chooses vector plots for metadata task `glu` or `glm`; otherwise it uses time-series plots. Use `--log-y` for time-series plots such as Ricker if desired.

## Python API Example

```python
import numpy as np

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

The test suite currently covers:

- RNG reproducibility
- prior sampling and log probabilities
- simulator shapes and reproducibility
- MCAR masks
- HDF5 contract validation and roundtrip behavior
- plotting helper behavior

Run tests with:

```bash
PYTHONPATH=src pytest
```

## Repository Layout

```text
configs/
scripts/
src/gapsbi/
  baselines/
  diagnostics/
  masks/
  simulators/
tests/
data/
```

The codebase is intentionally lightweight at this stage and uses NumPy-first implementations for dataset generation.