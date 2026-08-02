from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from gapsbi.simulators import GLMSimulator, LotkaVolterraSimulator


def _load_train_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "experiments" / "npe_full_data" / "train.py"
    spec = importlib.util.spec_from_file_location("npe_full_data_train", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_glm_tarp_references_use_true_prior_before_scaling() -> None:
    module = _load_train_module()
    scaler = StandardScaler().fit(GLMSimulator(summary="raw").sample_theta(256, np.random.default_rng(1)))
    actual = module.sample_tarp_references(
        problem="glm",
        num_references=8,
        seed=123,
        scaled_prior=torch.distributions.MultivariateNormal(torch.zeros(10), torch.eye(10)),
        theta_scaler=scaler,
        config={},
    ).numpy()
    expected_theta = GLMSimulator(summary="raw").sample_theta(8, np.random.default_rng(123))
    np.testing.assert_allclose(actual, scaler.transform(expected_theta).astype(np.float32))


def test_lotka_volterra_tarp_references_use_true_prior_before_scaling() -> None:
    module = _load_train_module()
    simulator = LotkaVolterraSimulator()
    scaler = StandardScaler().fit(simulator.sample_theta(256, np.random.default_rng(1)))
    actual = module.sample_tarp_references(
        problem="lotka_volterra",
        num_references=8,
        seed=123,
        scaled_prior=torch.distributions.MultivariateNormal(torch.zeros(4), torch.eye(4)),
        theta_scaler=scaler,
        config={},
    ).numpy()
    expected_theta = simulator.sample_theta(8, np.random.default_rng(123))
    np.testing.assert_allclose(actual, scaler.transform(expected_theta).astype(np.float32))
