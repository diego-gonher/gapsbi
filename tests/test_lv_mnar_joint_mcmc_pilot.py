"""Focused algebraic tests for the LV/MNAR joint-pilot implementation."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


_SPEC = importlib.util.spec_from_file_location(
    "lv_mnar_joint_pilot", Path("scripts/run_lv_mnar_joint_mcmc_pilot.py")
)
assert _SPEC and _SPEC.loader
pilot = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pilot)


def test_completion_from_mu_uses_standardized_missing_noise_only() -> None:
    x_obs = np.array([2.0, 3.0, 0.0, 0.0])
    mask = np.array([1, 1, 0, 0])
    mu = np.log(np.array([2.5, 3.5, 4.5, 5.5]))
    xi = np.array([1.0, -2.0])
    complete = pilot.completion_from_mu(x_obs, mask, mu, xi, sigma=.1)
    np.testing.assert_allclose(complete[:2], x_obs[:2])
    np.testing.assert_allclose(complete[2:], np.exp(mu[2:] + .1 * xi))


def test_elliptical_slice_handles_constant_finite_mask_likelihood(monkeypatch) -> None:
    # With a constant slice likelihood, ESS must be a pure Gaussian rotation;
    # this catches accidentally putting the Gaussian xi prior in the slice term.
    monkeypatch.setattr(pilot, "latent_log_likelihood", lambda *args, **kwargs: 0.0)
    result, value, evaluations = pilot.elliptical_slice_latent(
        np.array([.3, -.4]), np.zeros(4), np.ones(4), np.ones(4),
        simulator=SimpleNamespace(observation_noise_scale=.1), missing_fraction=.5, rng=np.random.default_rng(12),
        max_bracket_steps=5,
    )
    assert np.all(np.isfinite(result))
    assert value == 0.0
    assert evaluations == 1


def test_joint_target_has_one_standard_normal_latent_prior(monkeypatch) -> None:
    monkeypatch.setattr(pilot, "mean_log_observations", lambda z, simulator: np.zeros(4))
    monkeypatch.setattr(pilot, "lotka_volterra_log_prob", lambda *args, **kwargs: 7.0)
    monkeypatch.setattr(pilot, "latent_log_likelihood", lambda *args, **kwargs: -2.0)
    value, _ = pilot.joint_log_probability(
        np.zeros(4), np.array([1.0, -2.0]), np.ones(4), np.ones(4),
        SimpleNamespace(observation_noise_scale=.1), missing_fraction=.5,
    )
    assert value == 7.0 - 2.0 - .5 * (1.0 + 4.0)


def test_coupled_parameter_latent_move_preserves_completion_and_is_reversible() -> None:
    mask = np.array([1, 1, 0, 0])
    old_mu = np.log(np.array([2., 3., 4., 5.]))
    new_mu = np.log(np.array([2.5, 3.5, 6., 7.]))
    xi = np.array([.2, -.3])
    moved = pilot.coupled_latent_parameter_proposal(xi, old_mu, new_mu, mask, .1)
    x_obs = np.array([1.5, 2.5, 0., 0.])
    np.testing.assert_allclose(
        pilot.completion_from_mu(x_obs, mask, old_mu, xi, .1),
        pilot.completion_from_mu(x_obs, mask, new_mu, moved, .1),
    )
    np.testing.assert_allclose(
        pilot.coupled_latent_parameter_proposal(moved, new_mu, old_mu, mask, .1), xi,
    )

