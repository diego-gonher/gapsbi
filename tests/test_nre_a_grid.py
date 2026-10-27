from __future__ import annotations
import numpy as np
import torch
from gapsbi.methods.sbi_nre import evaluate_nre_a_oup_grid, sample_nre_a_oup_grid

class _Scaler:
    def transform(self, x): return np.asarray(x)
class _Ratio:
    def unnormalized_log_ratio(self, theta, x): return -theta.square().sum(dim=1)

def test_oup_grid_normalizes_and_samples_within_support():
    grid = evaluate_nre_a_oup_grid(_Ratio(), theta_scaler=_Scaler(), x_scaled=np.zeros(25), resolution=20, batch_size=100)
    assert np.isclose(grid["weights"].sum(), 1.0)
    assert np.isfinite(grid["mean"]).all() and np.isfinite(grid["cov"]).all()
    samples = sample_nre_a_oup_grid(grid, num_samples=100, seed=2)
    assert np.all(samples >= [0., -2.]) and np.all(samples <= [2., 3.])
