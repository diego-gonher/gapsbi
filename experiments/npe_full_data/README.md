# NPE Full-Data Experiment

This experiment reproduces the full-data notebook workflow from `reference.txt`
using reusable package utilities under `src/gapsbi`.

## What it does

- Loads a GAPSBI HDF5 dataset from `dataset_path`.
- Uses `x_full` only (not `x_obs` or `mask`).
- Uses predefined train/val/test splits from the dataset.
- Scales:
  - `theta` with `MinMaxScaler(feature_range=(-1, 1))` fit on train.
  - `x` with:
    - `log1p + StandardScaler` for Ricker.
    - `StandardScaler` for GLM/GLU/OUP.
- Builds a scaled-space `BoxUniform([-1, 1]^d)` prior.
- Runs NPE training/evaluation for each configured seed.
- Saves posterior samples, TARP/SBC diagnostics, and summary JSONs.

## Run

```bash
python experiments/npe_full_data/train.py --config experiments/npe_full_data/config.yaml
```

## Outputs

For each seed under `output_dir/seed_<seed>/`:

- `training_summary.png`
- `posterior_samples.h5`
- `tarp.png`
- `sbc_rank_histograms.png`
- `diagnostics_arrays.npz`
- `summary.json`

At the root `output_dir/`:

- `all_results.json`
