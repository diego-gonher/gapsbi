# NPE Full-Data Experiment

This experiment reproduces the full-data notebook workflow from `reference.txt`
using reusable package utilities under `src/gapsbi`.

## What it does

- Loads a GAPSBI HDF5 dataset from `dataset_path`.
- Uses `x_full` only (not `x_obs` or `mask`).
- Uses predefined train/val/test splits from the dataset.
- Scales:
  - `theta` with `StandardScaler` for GLM/Lotka-Volterra and `MinMaxScaler(feature_range=(-1, 1))` for bounded-prior tasks, fit on train.
  - `x` with:
    - `StandardScaler` for GLM/GLU/OUP/Lotka-Volterra.
    - `log1p + StandardScaler` only for legacy Ricker.
- Builds a scaled-space Gaussian prior with empirical train covariance for GLM/Lotka-Volterra and `BoxUniform([-1, 1]^d)` for bounded-prior tasks.
- Runs NPE training/evaluation for each configured seed.
- Saves posterior samples, TARP/SBC diagnostics, and summary JSONs.
- Optionally samples the trained posterior on the ten fixed reference
  observations when `reference_path` is configured.

## Run

```bash
python experiments/npe_full_data/train.py \
  --config experiments/npe_full_data/low_sim_budget/oup_config.yaml
```

Queue scripts are available for the three budget regimes:

```bash
bash experiments/npe_full_data/npe_full_data_run_queue_high_sim_budget.sh
bash experiments/npe_full_data/npe_full_data_run_queue_mid_sim_budget.sh
bash experiments/npe_full_data/npe_full_data_run_queue_low_sim_budget.sh
```

## Outputs

For each seed under `output_dir/seed_<seed>/`:

- `training_summary.png`
- `posterior_samples.h5`
- `reference_posterior_samples.h5` when `reference_path` is configured
- `tarp.png`
- `sbc_rank_histograms.png`
- `diagnostics_arrays.npz`
- `summary.json`

At the root `output_dir/`:

- `all_results.json`
