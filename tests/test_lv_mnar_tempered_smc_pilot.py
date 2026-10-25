"""Numerical checks for the LV/MNAR tempered-SMC bridge helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


_SPEC = importlib.util.spec_from_file_location(
    "lv_mnar_tempered_smc", Path("scripts/run_lv_mnar_tempered_smc_pilot.py")
)
assert _SPEC and _SPEC.loader
smc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smc)


def test_incremental_tempering_weights_and_ess() -> None:
    log_likelihood = np.array([-2.0, -1.0, 0.0])
    beta_old, beta_new = .2, .7
    weights, ess, _ = smc.normalize((beta_new - beta_old) * log_likelihood)
    expected = np.exp((beta_new - beta_old) * log_likelihood)
    expected /= expected.sum()
    np.testing.assert_allclose(weights, expected)
    assert np.isclose(ess, 1.0 / np.sum(expected**2))


def test_adaptive_beta_advances_or_reaches_one() -> None:
    beta = smc.choose_beta(0.0, np.array([-3.0, -1.0, 0.0, 1.0]), np.zeros(4), 4)
    assert 0.0 < beta <= 1.0
