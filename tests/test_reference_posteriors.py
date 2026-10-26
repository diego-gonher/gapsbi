from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm, truncnorm

from gapsbi.references import (
    PARTIAL_DATA_REFERENCE_SCHEMA_NAME,
    REFERENCE_SCHEMA_NAME,
    default_full_reference_path,
    default_partial_reference_path,
    compute_oup_grid_posterior,
    generate_glm_reference_posteriors,
    generate_glu_reference_posteriors,
    generate_glu_mcar_partial_data_reference_posteriors,
    generate_lotka_volterra_reference_posteriors,
    generate_oup_reference_posteriors,
    generate_oup_mcar_partial_data_reference_posteriors,
    glm_log_likelihood,
    glm_mnar_compatible_completions,
    glm_mnar_log_marginal_likelihood,
    glm_mnar_mask_log_probability,
    glm_observed_laplace_covariance,
    glm_observed_log_likelihood,
    glm_observed_log_prob,
    find_glm_observed_map,
    run_glm_emcee_reference,
    run_glm_observed_independence_mh,
    load_reference_posteriors_hdf5,
    load_glu_mcar_partial_data_reference_posteriors_hdf5,
    load_partial_data_reference_posteriors_hdf5,
    lotka_volterra_log_likelihood,
    lotka_volterra_log_prob,
    lotka_volterra_log_total_mnar_log_probability,
    lotka_volterra_log_total_mnar_soft_log_probability,
    sample_lotka_volterra_mnar_missing_observations,
    find_lotka_volterra_map,
    initialize_screened_lotka_volterra_walkers,
    plot_lotka_volterra_reference_predictives,
    oup_log_likelihood,
    oup_mcar_observed_log_likelihood,
    oup_mnar_mask_log_probability,
    sample_oup_conditional_gaussian_trajectories,
    sample_oup_mnar_importance,
    plot_reference_mcmc_traces,
    plot_reference_posterior_marginals,
    plot_reference_posterior_pairs,
    save_reference_posteriors_hdf5,
    save_glu_mcar_partial_data_reference_posteriors_hdf5,
    save_partial_data_reference_posteriors_hdf5,
    sample_glu_reference_posterior,
    sample_glu_mcar_partial_data_reference_posterior,
    glu_mnar_mask_log_probability,
)
from gapsbi.masks import LotkaVolterraLogTotalMNARMask, LotkaVolterraTimeBlockMCARMask, PointMCARMask
from gapsbi.simulators import GLMSimulator, LotkaVolterraSimulator, OUPSimulator


def test_glm_mnar_constant_zero_completion_is_not_discarded() -> None:
    x_obs = np.array([0.0, 0.0, 0.0, 0.0])
    mask = np.array([1, 1, 0, 0], dtype=np.int8)
    completions, log_mask = glm_mnar_compatible_completions(
        x_obs, mask, missing_fraction=0.5
    )
    assert completions.shape == (2, 4)
    assert {tuple(row) for row in completions} == {(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 1.0)}
    assert np.all(np.isfinite(log_mask))


def test_reference_path_helpers_preserve_versioned_artifact_names() -> None:
    assert default_full_reference_path("glu") == Path("references/reference_posteriors_v1/glu_references.h5")
    assert default_partial_reference_path("oup", "mnar", .25) == Path(
        "references/partial_data_references_v1/oup_mnar_eps025_references.h5"
    )
    assert default_partial_reference_path("lotka_volterra", "mnar", .5) == Path(
        "references/partial_data_references_v1/lotka_volterra_mnar_eps050_successful_references.h5"
    )
    assert default_full_reference_path("glm", root="references/full") == Path("references/full/glm_references.h5")


def test_glm_mnar_mask_probability_matches_generator() -> None:
    from gapsbi.masks import MeanNormalizedSelfCensoringMNARMask
    x = np.array([0.0, 0.0, 1.0, 1.0])
    target = np.array([1, 1, 0, 1], dtype=np.int8)
    probability = np.exp(glm_mnar_mask_log_probability(x, target, missing_fraction=0.5))
    masks = MeanNormalizedSelfCensoringMNARMask(0.5).generate(
        np.tile(x, (100_000, 1)), None, np.random.default_rng(17)
    )
    np.testing.assert_allclose(np.mean(np.all(masks == target, axis=1)), probability, atol=0.003)


def test_glm_mnar_marginal_likelihood_matches_brute_force_binary_enumeration() -> None:
    simulator = GLMSimulator(summary="raw")
    x_obs = np.zeros(100)
    mask = np.ones(100, dtype=np.int8)
    mask[:2] = 0
    theta = np.zeros(simulator.theta_dim)
    exact = glm_mnar_log_marginal_likelihood(theta, x_obs, mask, missing_fraction=0.5, simulator=simulator)
    terms = []
    for bits in ((0, 0), (0, 1), (1, 0), (1, 1)):
        full = x_obs.copy(); full[:2] = bits
        lp = glm_mnar_mask_log_probability(full, mask, missing_fraction=0.5)
        if np.isfinite(lp):
            terms.append(glm_log_likelihood(theta, full, simulator) + lp)
    np.testing.assert_allclose(exact, __import__('scipy').special.logsumexp(terms))


def test_generate_glu_reference_posteriors_contract() -> None:
    observations, metadata, reference_data = generate_glu_reference_posteriors(
        num_observations=3,
        num_reference_samples=200,
        observation_seed=123,
        posterior_seed=456,
        dim=4,
    )

    theta_samples = reference_data["theta_samples"]
    assert observations["theta_true"].shape == (3, 4)
    assert observations["x_full"].shape == (3, 4)
    assert theta_samples.shape == (3, 200, 4)
    assert metadata["problem"] == "glu"
    assert metadata["theta_scaled"] is False
    assert metadata["x_scaled"] is False
    assert reference_data["num_prior_bound_violations"] == 0
    assert np.all(theta_samples >= -1.0)
    assert np.all(theta_samples <= 1.0)


def test_glu_reference_sampler_concentrates_near_observation() -> None:
    rng = np.random.default_rng(123)
    x_full = np.array([[0.0, 0.5]])
    samples = sample_glu_reference_posterior(
        x_full=x_full,
        prior_low=np.array([-1.0, -1.0]),
        prior_high=np.array([1.0, 1.0]),
        simulator_scale=0.1,
        num_reference_samples=20_000,
        rng=rng,
    )

    assert samples.shape == (1, 20_000, 2)
    np.testing.assert_allclose(samples[0].mean(axis=0), x_full[0], atol=0.01)


def test_glu_mcar_fully_observed_reproduces_full_data_sampler() -> None:
    kwargs = {
        "x_obs": np.array([[0.1, -0.2]]),
        "prior_low": np.array([-1.0, -1.0]),
        "prior_high": np.array([1.0, 1.0]),
        "simulator_scale": 0.1,
        "num_reference_samples": 100,
    }
    partial = sample_glu_mcar_partial_data_reference_posterior(
        mask=np.ones((1, 2), dtype=np.int8), rng=np.random.default_rng(123), **kwargs
    )
    full = sample_glu_reference_posterior(
        x_full=kwargs["x_obs"], rng=np.random.default_rng(123), **{
            key: value for key, value in kwargs.items() if key != "x_obs"
        }
    )
    np.testing.assert_array_equal(partial, full)


def test_glu_mcar_fully_missing_recovers_uniform_prior() -> None:
    samples = sample_glu_mcar_partial_data_reference_posterior(
        x_obs=np.zeros((1, 2)),
        mask=np.zeros((1, 2), dtype=np.int8),
        prior_low=np.array([-2.0, -4.0]),
        prior_high=np.array([2.0, 4.0]),
        simulator_scale=0.1,
        num_reference_samples=40_000,
        rng=np.random.default_rng(123),
    )
    np.testing.assert_allclose(samples[0].mean(axis=0), [0.0, 0.0], atol=0.04)
    assert np.all(samples >= [-2.0, -4.0])
    assert np.all(samples <= [2.0, 4.0])


def test_glu_mcar_mixed_mask_has_coordinatewise_posterior() -> None:
    samples = sample_glu_mcar_partial_data_reference_posterior(
        x_obs=np.array([[0.25, 0.0]]),
        mask=np.array([[1, 0]], dtype=np.int8),
        prior_low=np.array([-1.0, -1.0]),
        prior_high=np.array([1.0, 1.0]),
        simulator_scale=0.1,
        num_reference_samples=40_000,
        rng=np.random.default_rng(321),
    )
    np.testing.assert_allclose(samples[0, :, 0].mean(), 0.25, atol=0.01)
    np.testing.assert_allclose(samples[0, :, 1].mean(), 0.0, atol=0.02)
    assert np.all(samples >= -1.0)
    assert np.all(samples <= 1.0)


def test_glu_mcar_samples_match_known_coordinatewise_moments() -> None:
    x_obs = np.array([[0.3, 0.0]])
    prior_low = np.array([-1.0, -2.0])
    prior_high = np.array([1.0, 2.0])
    scale = 0.4
    samples = sample_glu_mcar_partial_data_reference_posterior(
        x_obs=x_obs,
        mask=np.array([[1, 0]], dtype=np.int8),
        prior_low=prior_low,
        prior_high=prior_high,
        simulator_scale=scale,
        num_reference_samples=100_000,
        rng=np.random.default_rng(99),
    )[0]
    a = (prior_low[0] - x_obs[0, 0]) / scale
    b = (prior_high[0] - x_obs[0, 0]) / scale
    expected_mean, expected_variance = truncnorm.stats(
        a, b, loc=x_obs[0, 0], scale=scale, moments="mv"
    )
    np.testing.assert_allclose(samples[:, 0].mean(), expected_mean, atol=0.004)
    np.testing.assert_allclose(samples[:, 0].var(), expected_variance, atol=0.003)
    np.testing.assert_allclose(samples[:, 1].mean(), 0.0, atol=0.01)
    np.testing.assert_allclose(samples[:, 1].var(), (4.0**2) / 12.0, atol=0.01)


def test_glu_mnar_mask_log_probability_matches_empirical_frequency() -> None:
    from gapsbi.masks import MeanNormalizedSelfCensoringMNARMask
    x = np.array([0.0, 0.5, 1.0])
    target = np.array([1, 1, 0], dtype=np.int8)
    eps = 0.5
    analytic = np.exp(glu_mnar_mask_log_probability(x, target, missing_fraction=eps))
    rng = np.random.default_rng(123)
    masks = MeanNormalizedSelfCensoringMNARMask(eps).generate(np.tile(x, (100_000, 1)), None, rng)
    empirical = np.mean(np.all(masks == target, axis=1))
    np.testing.assert_allclose(empirical, analytic, atol=0.003)


def test_glu_mnar_impossible_minimum_missing_has_minus_infinity_log_probability() -> None:
    assert glu_mnar_mask_log_probability(
        np.array([0.0, 0.5, 1.0]), np.array([0, 1, 1]), missing_fraction=0.5
    ) == -np.inf


def test_glu_mcar_partial_reference_roundtrip_and_evaluation_mask(tmp_path: Path) -> None:
    observations, metadata, reference_data = generate_glu_mcar_partial_data_reference_posteriors(
        num_observations=3,
        num_reference_samples=100,
        observation_seed=123,
        posterior_seed=456,
        dim=4,
        missing_fraction=0.25,
        reference_mask_seed=987_654,
    )
    expected_mask = PointMCARMask(0.25).generate(
        observations["x_full"], observations["theta_true"], np.random.default_rng(987_654)
    )
    np.testing.assert_array_equal(observations["mask"], expected_mask)
    np.testing.assert_array_equal(observations["x_obs"], observations["x_full"] * expected_mask)

    path = tmp_path / "glu_mcar_partial_references.h5"
    theta_samples = reference_data.pop("theta_samples")
    save_partial_data_reference_posteriors_hdf5(
        path,
        observations=observations,
        theta_samples=theta_samples,
        metadata=metadata,
        diagnostics=reference_data,
    )
    loaded_observations, loaded_samples, loaded_metadata, _ = (
        load_partial_data_reference_posteriors_hdf5(path)
    )
    for key, value in observations.items():
        np.testing.assert_array_equal(loaded_observations[key], value)
    np.testing.assert_array_equal(loaded_samples, theta_samples)
    assert loaded_metadata["schema_name"] == PARTIAL_DATA_REFERENCE_SCHEMA_NAME
    assert loaded_metadata["reference_mask_seed"] == 987_654


def test_save_load_reference_posteriors_roundtrip(tmp_path: Path) -> None:
    observations, metadata, reference_data = generate_glu_reference_posteriors(
        num_observations=2,
        num_reference_samples=100,
        observation_seed=123,
        posterior_seed=456,
        dim=3,
    )
    theta_samples = reference_data.pop("theta_samples")
    path = tmp_path / "glu_references.h5"

    save_reference_posteriors_hdf5(
        path,
        observations=observations,
        theta_samples=theta_samples,
        metadata=metadata,
        diagnostics=reference_data,
    )
    loaded_observations, loaded_samples, loaded_metadata, loaded_diagnostics = (
        load_reference_posteriors_hdf5(path)
    )

    np.testing.assert_array_equal(loaded_observations["theta_true"], observations["theta_true"])
    np.testing.assert_array_equal(loaded_observations["x_full"], observations["x_full"])
    np.testing.assert_array_equal(loaded_samples, theta_samples)
    assert loaded_metadata["schema_name"] == REFERENCE_SCHEMA_NAME
    assert loaded_metadata["problem"] == "glu"
    assert loaded_metadata["theta_scaled"] is False
    assert loaded_metadata["simulator"]["name"] == "glu"
    assert "posterior_sample_mean" in loaded_diagnostics


def test_save_reference_refuses_overwrite_by_default(tmp_path: Path) -> None:
    observations, metadata, reference_data = generate_glu_reference_posteriors(
        num_observations=1,
        num_reference_samples=10,
        dim=2,
    )
    theta_samples = reference_data.pop("theta_samples")
    path = tmp_path / "glu_references.h5"

    save_reference_posteriors_hdf5(
        path,
        observations=observations,
        theta_samples=theta_samples,
        metadata=metadata,
    )
    with pytest.raises(FileExistsError):
        save_reference_posteriors_hdf5(
            path,
            observations=observations,
            theta_samples=theta_samples,
            metadata=metadata,
        )


def test_generate_reference_posteriors_cli(tmp_path: Path) -> None:
    output_path = tmp_path / "glu_references.h5"
    repo_root = Path(__file__).parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_reference_posteriors.py",
            "--problem",
            "glu",
            "--num-observations",
            "2",
            "--num-reference-samples",
            "50",
            "--dim",
            "3",
            "--output",
            str(output_path),
        ],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Saved:" in result.stdout
    observations, theta_samples, metadata, _ = load_reference_posteriors_hdf5(output_path)
    assert observations["x_full"].shape == (2, 3)
    assert theta_samples.shape == (2, 50, 3)
    assert metadata["problem"] == "glu"


def test_plot_reference_posteriors_saves_one_plot_per_observation(tmp_path: Path) -> None:
    observations, metadata, reference_data = generate_glu_reference_posteriors(
        num_observations=2,
        num_reference_samples=100,
        observation_seed=123,
        posterior_seed=456,
        dim=3,
    )
    theta_samples = reference_data.pop("theta_samples")
    reference_path = tmp_path / "glu_references.h5"
    save_reference_posteriors_hdf5(
        reference_path,
        observations=observations,
        theta_samples=theta_samples,
        metadata=metadata,
        diagnostics=reference_data,
    )

    paths = plot_reference_posterior_marginals(reference_path, max_samples=50)

    assert len(paths) == 2
    for path in paths:
        assert path.exists()
        assert path.parent == tmp_path / "plots"


def test_oup_log_likelihood_matches_hand_computed_transition() -> None:
    simulator = OUPSimulator(n=2, T=5.0, var=0.1, y0=10.0)
    theta = np.array([0.5, 0.0])
    theta2 = np.exp(theta[1])
    mean_next = simulator.y0 + theta[0] * (theta2 - simulator.y0) * simulator.dt
    x = np.array([simulator.y0, mean_next])
    noise_var = 0.25 * simulator.dt * simulator.var

    actual = oup_log_likelihood(theta, x, simulator)
    expected = -0.5 * np.log(2.0 * np.pi * noise_var)

    np.testing.assert_allclose(actual, expected)


def test_oup_mcar_observed_likelihood_recovers_full_data_and_masked_initial() -> None:
    simulator = OUPSimulator(n=5, T=5.0, var=0.1, y0=10.0)
    theta = np.array([[0.0, 0.0], [0.7, -0.3], [1.5, 0.8]])
    x = np.array([10.0, 9.1, 8.4, 7.6, 7.2])
    full = np.ones(simulator.n, dtype=np.int8)
    initial_missing = full.copy()
    initial_missing[0] = 0
    expected = oup_log_likelihood(theta, x, simulator)
    np.testing.assert_allclose(
        oup_mcar_observed_log_likelihood(theta, x, full, simulator), expected
    )
    np.testing.assert_allclose(
        oup_mcar_observed_log_likelihood(theta, x, initial_missing, simulator), expected
    )


def test_oup_mcar_observed_likelihood_matches_k_step_transition_and_quadrature() -> None:
    simulator = OUPSimulator(n=4, T=4.0, var=0.2, y0=10.0)
    theta = np.array([0.6, np.log(2.0)])
    x = np.array([10.0, -999.0, -999.0, 7.5])
    mask = np.array([1, 0, 0, 1], dtype=np.int8)
    a = 1.0 - theta[0] * simulator.dt
    b = theta[0] * np.exp(theta[1]) * simulator.dt
    q = 0.25 * simulator.dt * simulator.var
    mean = a**3 * simulator.y0 + b * sum(a**r for r in range(3))
    variance = q * sum(a ** (2 * r) for r in range(3))
    expected = -0.5 * ((x[3] - mean) ** 2 / variance + np.log(2.0 * np.pi * variance))
    actual = oup_mcar_observed_log_likelihood(theta, x, mask, simulator)
    np.testing.assert_allclose(actual, expected)

    # Independently integrate out the two hidden Euler states.
    def one_step(next_value: float, current_value: float) -> float:
        transition_mean = a * current_value + b
        return np.exp(-0.5 * ((next_value - transition_mean) ** 2 / q + np.log(2.0 * np.pi * q)))

    def marginal_given_first(z1: float) -> float:
        inner = quad(
            lambda z2: one_step(z2, z1) * one_step(x[3], z2),
            -np.inf, np.inf, epsabs=1e-120,
        )[0]
        return one_step(z1, simulator.y0) * inner

    numerical = quad(marginal_given_first, -np.inf, np.inf, epsabs=1e-120)[0]
    np.testing.assert_allclose(np.log(numerical), actual, rtol=1e-8, atol=1e-8)


def test_oup_mcar_observed_likelihood_handles_kappa_zero_and_edge_masks() -> None:
    simulator = OUPSimulator(n=5, T=5.0, var=0.1, y0=10.0)
    theta = np.array([0.0, 2.0])
    x = np.array([123.0, -3.0, 8.0, -4.0, 9.0])
    # Initial and intermediate values are masked; fixed y0 anchors the 2-step transition.
    mask = np.array([0, 0, 1, 0, 1], dtype=np.int8)
    q = 0.25 * simulator.dt * simulator.var
    expected = -0.5 * ((x[2] - simulator.y0) ** 2 / (2 * q) + np.log(2.0 * np.pi * 2 * q))
    expected += -0.5 * ((x[4] - x[2]) ** 2 / (2 * q) + np.log(2.0 * np.pi * 2 * q))
    np.testing.assert_allclose(
        oup_mcar_observed_log_likelihood(theta, x, mask, simulator), expected
    )
    # No observed states leaves the likelihood constant (the posterior is the prior).
    np.testing.assert_allclose(
        oup_mcar_observed_log_likelihood(theta, np.zeros(5), np.zeros(5, dtype=np.int8), simulator),
        0.0,
    )
    # A single observed state after time zero is a single multi-step factor.
    single_mask = np.array([0, 0, 0, 1, 0], dtype=np.int8)
    single = oup_mcar_observed_log_likelihood(theta, x, single_mask, simulator)
    expected_single = -0.5 * ((x[3] - simulator.y0) ** 2 / (3 * q) + np.log(2.0 * np.pi * 3 * q))
    np.testing.assert_allclose(single, expected_single)


def test_oup_mcar_empty_mask_grid_is_uniform_prior() -> None:
    simulator = OUPSimulator(n=4)
    posterior = compute_oup_grid_posterior(
        np.zeros(simulator.n), simulator=simulator, grid_resolution=30,
        mask=np.zeros(simulator.n, dtype=np.int8),
    )
    np.testing.assert_allclose(posterior["weights"], 1.0 / (30 * 30))
    np.testing.assert_allclose(posterior["mean"], np.array([1.0, 0.5]))


def test_oup_mnar_mask_log_probability_matches_generator_and_clipping() -> None:
    from gapsbi.masks import MeanNormalizedSelfCensoringMNARMask
    x = np.array([10.0, 4.0, 7.0, 9.0])
    mask = np.array([1, 1, 1, 0], dtype=np.int8)
    probability = np.exp(oup_mnar_mask_log_probability(x, mask, missing_fraction=0.25))
    masks = MeanNormalizedSelfCensoringMNARMask(0.25).generate(
        np.tile(x, (100_000, 1)), None, np.random.default_rng(3)
    )
    np.testing.assert_allclose(np.mean(np.all(masks == mask, axis=1)), probability, atol=0.003)
    # The minimum has zero missingness probability under identity scoring.
    assert oup_mnar_mask_log_probability(
        x, np.array([1, 0, 1, 1], dtype=np.int8), missing_fraction=0.5
    ) == -np.inf


def test_oup_conditional_gaussian_bridge_moments_and_masked_initial() -> None:
    simulator = OUPSimulator(n=4, T=4.0, var=0.2, y0=10.0)
    theta = np.array([0.6, np.log(2.0)])
    x_obs = np.array([10.0, 0.0, 0.0, 7.5])
    mask = np.array([0, 0, 0, 1], dtype=np.int8)
    draws = sample_oup_conditional_gaussian_trajectories(
        np.repeat(theta[None, :], 50_000, axis=0), x_obs, mask,
        simulator=simulator, rng=np.random.default_rng(8),
    )
    assert np.all(draws[:, 0] == simulator.y0)
    np.testing.assert_allclose(draws[:, 3], x_obs[3])
    a = 1 - theta[0] * simulator.dt
    b = theta[0] * np.exp(theta[1]) * simulator.dt
    q = 0.25 * simulator.dt * simulator.var
    # Joint Gaussian moments for (Y1, Y2, Y3), conditioned on Y3=7.5.
    mean = np.array([a * simulator.y0 + b, a**2 * simulator.y0 + b * (1 + a),
                     a**3 * simulator.y0 + b * (1 + a + a**2)])
    cov = np.array([[q, a*q, a*a*q], [a*q, q*(1+a*a), a*q*(1+a*a)],
                    [a*a*q, a*q*(1+a*a), q*(1+a*a+a**4)]])
    conditioned_mean = mean[:2] + cov[:2, 2] / cov[2, 2] * (x_obs[3] - mean[2])
    conditioned_cov = cov[:2, :2] - np.outer(cov[:2, 2], cov[2, :2]) / cov[2, 2]
    np.testing.assert_allclose(draws[:, 1:3].mean(axis=0), conditioned_mean, atol=0.01)
    np.testing.assert_allclose(np.cov(draws[:, 1:3], rowvar=False), conditioned_cov, atol=0.003)


def test_oup_mnar_importance_constant_mask_factor_recovers_mcar_proposal() -> None:
    simulator = OUPSimulator(n=4)
    x = simulator.simulate(np.array([0.4, 0.2]), np.random.default_rng(10))
    mcar = compute_oup_grid_posterior(x, simulator=simulator, grid_resolution=30)
    samples, diagnostics = sample_oup_mnar_importance(
        x_obs=x, mask=np.ones(simulator.n, dtype=np.int8), missing_fraction=0.0,
        mcar_posterior=mcar, num_proposals=1_000, num_samples=100,
        rng=np.random.default_rng(11), simulator=simulator, batch_size=200,
    )
    assert samples.shape == (100, 2)
    assert diagnostics["ess"] == pytest.approx(1_000.0)
    assert diagnostics["zero_weight_fraction"] == 0.0


def test_oup_mnar_short_trajectory_mask_integral_matches_quadrature() -> None:
    simulator = OUPSimulator(n=2, T=2.0, var=0.1, y0=10.0)
    theta = np.array([0.0, 0.0])
    mask = np.array([1, 0], dtype=np.int8)
    q = 0.25 * simulator.dt * simulator.var
    # For kappa=0, Y1 ~ N(10,q); with this mask the exact likelihood is
    # 2*f on Y1>10 and zero on Y1<10 (the equality event has zero measure).
    def integrand(y1: float) -> float:
        return np.exp(oup_mnar_mask_log_probability(
            np.array([simulator.y0, y1]), mask, missing_fraction=0.25
        )) * np.exp(-0.5 * ((y1 - simulator.y0) ** 2 / q + np.log(2 * np.pi * q)))
    # Split at the score discontinuity (the trajectory minimum switches there).
    integral = quad(integrand, simulator.y0, np.inf)[0]
    np.testing.assert_allclose(integral, 0.25, atol=1e-6)


def test_generate_oup_mcar_partial_references_uses_full_artifact_and_mask(tmp_path: Path) -> None:
    observations, metadata, reference_data = generate_oup_reference_posteriors(
        num_observations=2, num_reference_samples=20, observation_seed=123,
        posterior_seed=456, grid_resolution=20, validation_grid_resolutions=(),
        spotcheck_grid_resolution=None,
    )
    source = tmp_path / "oup_full.h5"
    save_reference_posteriors_hdf5(
        source, observations=observations, theta_samples=reference_data["theta_samples"],
        metadata=metadata, diagnostics={key: value for key, value in reference_data.items() if key != "theta_samples"},
    )
    partial_observations, partial_metadata, partial_data = (
        generate_oup_mcar_partial_data_reference_posteriors(
            full_reference_path=source, missing_fraction=0.25, posterior_seed=789,
            num_reference_samples=30, grid_resolution=24, validation_grid_resolution=12,
            validation_observation_indices=(0,),
        )
    )
    expected_mask = PointMCARMask(0.25).generate(
        observations["x_full"], observations["theta_true"], np.random.default_rng(987_654)
    )
    np.testing.assert_array_equal(partial_observations["theta_true"], observations["theta_true"])
    np.testing.assert_array_equal(partial_observations["x_full"], observations["x_full"])
    np.testing.assert_array_equal(partial_observations["mask"], expected_mask)
    np.testing.assert_array_equal(partial_observations["x_obs"], observations["x_full"] * expected_mask)
    assert partial_data["theta_samples"].shape == (2, 30, 2)
    assert np.all(np.isfinite(partial_data["theta_samples"]))
    assert partial_data["num_prior_bound_violations"] == 0
    assert partial_metadata["posterior_method"] == "deterministic_grid_exact_euler_observed_data_likelihood"


def test_generate_oup_reference_posteriors_contract_small_grid() -> None:
    observations, metadata, reference_data = generate_oup_reference_posteriors(
        num_observations=2,
        num_reference_samples=100,
        observation_seed=123,
        posterior_seed=456,
        grid_resolution=40,
        validation_grid_resolutions=(20,),
        spotcheck_grid_resolution=None,
    )

    theta_samples = reference_data["theta_samples"]
    assert observations["theta_true"].shape == (2, 2)
    assert observations["x_full"].shape == (2, 25)
    assert theta_samples.shape == (2, 100, 2)
    assert metadata["problem"] == "oup"
    assert metadata["posterior_method"] == "deterministic_grid"
    assert reference_data["grid_mean"].shape == (2, 2)
    assert reference_data["grid_cov"].shape == (2, 2, 2)
    assert reference_data["validation_mean_l2_delta_vs_final_grid"].shape == (1, 2)
    assert reference_data["num_prior_bound_violations"] == 0
    assert np.all(theta_samples[:, :, 0] >= 0.0)
    assert np.all(theta_samples[:, :, 0] <= 2.0)
    assert np.all(theta_samples[:, :, 1] >= -2.0)
    assert np.all(theta_samples[:, :, 1] <= 2.0)


def test_oup_grid_posterior_normalizes() -> None:
    simulator = OUPSimulator(n=4)
    rng = np.random.default_rng(123)
    theta = np.array([0.5, 0.0])
    x = simulator.simulate(theta, rng)

    posterior = compute_oup_grid_posterior(
        x,
        simulator=simulator,
        grid_resolution=25,
    )

    np.testing.assert_allclose(np.sum(posterior["weights"]), 1.0)
    assert posterior["ess"] > 1.0
    assert np.asarray(posterior["mean"]).shape == (2,)


def test_plot_reference_pair_saves_for_two_dimensional_references(tmp_path: Path) -> None:
    observations, metadata, reference_data = generate_oup_reference_posteriors(
        num_observations=1,
        num_reference_samples=100,
        observation_seed=123,
        posterior_seed=456,
        grid_resolution=30,
        validation_grid_resolutions=(),
        spotcheck_grid_resolution=None,
    )
    theta_samples = reference_data.pop("theta_samples")
    reference_path = tmp_path / "oup_references.h5"
    save_reference_posteriors_hdf5(
        reference_path,
        observations=observations,
        theta_samples=theta_samples,
        metadata=metadata,
        diagnostics=reference_data,
    )

    paths = plot_reference_posterior_pairs(reference_path, max_samples=50)

    assert len(paths) == 1
    assert paths[0].exists()


def test_glm_log_likelihood_matches_hand_computed_terms() -> None:
    simulator = GLMSimulator(dim=2, duration=3, summary="raw")
    theta = np.array([0.25, -0.5])
    x = np.array([0, 1, 1])
    psi = theta @ simulator.design_matrix.T
    expected = np.sum(x * psi - np.logaddexp(0.0, psi))

    actual = glm_log_likelihood(theta, x, simulator)

    np.testing.assert_allclose(actual, expected)


def test_glm_observed_log_likelihood_restricts_to_masked_rows() -> None:
    simulator = GLMSimulator(dim=3, duration=6, summary="raw")
    theta = np.array([[0.25, -0.5, 0.1], [-0.1, 0.2, 0.3]])
    x_obs = np.array([0, 1, 0, 1, 0, 1])
    mask = np.array([1, 0, 1, 0, 0, 1], dtype=np.int8)
    design = simulator.design_matrix[mask.astype(bool)]
    x = x_obs[mask.astype(bool)]
    expected = np.sum(x[None, :] * (theta @ design.T) - np.logaddexp(0.0, theta @ design.T), axis=1)
    np.testing.assert_allclose(glm_observed_log_likelihood(theta, x_obs, mask, simulator), expected)
    # Missing entries are not inspected, so arbitrary imputations cannot alter the target.
    x_with_nonbinary_missing = x_obs.copy()
    x_with_nonbinary_missing[mask == 0] = 17
    np.testing.assert_allclose(
        glm_observed_log_likelihood(theta, x_with_nonbinary_missing, mask, simulator), expected
    )


def test_glm_observed_full_and_empty_masks_recover_expected_targets() -> None:
    simulator = GLMSimulator(dim=3, duration=8, summary="raw")
    theta = np.array([[0.2, -0.4, 0.3], [-0.1, 0.5, -0.2]])
    x = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=np.int8)
    full_mask = np.ones(8, dtype=np.int8)
    empty_mask = np.zeros(8, dtype=np.int8)
    np.testing.assert_allclose(
        glm_observed_log_likelihood(theta, x, full_mask, simulator),
        glm_log_likelihood(theta, x, simulator),
    )
    np.testing.assert_allclose(
        glm_observed_log_likelihood(theta, np.zeros_like(x), empty_mask, simulator), 0.0,
    )
    np.testing.assert_allclose(
        glm_observed_log_prob(theta, np.zeros_like(x), empty_mask, simulator), simulator.log_prior(theta),
    )
    mode, _ = find_glm_observed_map(np.zeros_like(x), empty_mask, simulator)
    np.testing.assert_allclose(mode, simulator.prior_mean, atol=1e-7)
    np.testing.assert_allclose(
        glm_observed_laplace_covariance(mode, np.zeros_like(x), empty_mask, simulator),
        simulator.prior_covariance,
    )


def test_glm_observed_independence_mh_empty_mask_samples_gaussian_prior() -> None:
    simulator = GLMSimulator(dim=3, duration=8, summary="raw")
    result = run_glm_observed_independence_mh(
        np.zeros(8, dtype=np.int8), np.zeros(8, dtype=np.int8), simulator=simulator,
        num_steps=12_000, burn_in_steps=1_000, thin=2, rng=np.random.default_rng(11),
    )
    samples = result["chain"]
    assert result["acceptance_fraction"] > 0.999
    np.testing.assert_allclose(samples.mean(axis=0), simulator.prior_mean, atol=0.08)
    np.testing.assert_allclose(np.cov(samples, rowvar=False), simulator.prior_covariance, atol=0.12)


def test_generate_glm_reference_posteriors_contract_short_mcmc() -> None:
    observations, metadata, reference_data = generate_glm_reference_posteriors(
        num_observations=1,
        num_reference_samples=20,
        observation_seed=123,
        posterior_seed=456,
        dim=3,
        duration=12,
        num_walkers=12,
        burn_in_steps=10,
        production_steps=12,
        trace_num_steps=6,
        trace_num_walkers=3,
        validation_num_ensembles=0,
        validation_observation_indices=(),
    )

    theta_samples = reference_data["theta_samples"]
    assert observations["theta_true"].shape == (1, 3)
    assert observations["x_full"].shape == (1, 12)
    assert theta_samples.shape == (1, 20, 3)
    assert metadata["problem"] == "glm"
    assert metadata["posterior_method"] == "emcee_ensemble_mcmc"
    assert reference_data["acceptance_fraction"].shape == (1,)
    assert reference_data["split_rhat"].shape == (1, 3)
    assert reference_data["mcmc_trace"].shape == (1, 6, 3, 3)
    assert reference_data["num_nonfinite_prior_log_prob"] == 0
    assert np.all(np.isfinite(theta_samples))


def test_glm_emcee_seed_controls_internal_sampler_randomness() -> None:
    simulator = GLMSimulator(dim=3, duration=12, summary="raw")
    x_full = simulator.simulate(np.zeros(3), np.random.default_rng(7))
    settings = dict(simulator=simulator, num_walkers=12, burn_in_steps=8,
                    production_steps=10, initial_scale=.05, emcee_seed=345)
    first = run_glm_emcee_reference(x_full, rng=np.random.default_rng(123), **settings)
    second = run_glm_emcee_reference(x_full, rng=np.random.default_rng(123), **settings)
    np.testing.assert_array_equal(first["chain"], second["chain"])


def test_plot_reference_mcmc_traces_saves_when_available(tmp_path: Path) -> None:
    observations, metadata, reference_data = generate_glm_reference_posteriors(
        num_observations=1,
        num_reference_samples=20,
        observation_seed=123,
        posterior_seed=456,
        dim=3,
        duration=12,
        num_walkers=12,
        burn_in_steps=10,
        production_steps=12,
        trace_num_steps=6,
        trace_num_walkers=3,
        validation_num_ensembles=0,
        validation_observation_indices=(),
    )
    theta_samples = reference_data.pop("theta_samples")
    reference_path = tmp_path / "glm_references.h5"
    save_reference_posteriors_hdf5(
        reference_path,
        observations=observations,
        theta_samples=theta_samples,
        metadata=metadata,
        diagnostics=reference_data,
    )

    paths = plot_reference_mcmc_traces(reference_path)

    assert len(paths) == 1
    assert paths[0].exists()


def test_lotka_volterra_log_likelihood_is_finite_at_noiseless_trajectory() -> None:
    simulator = LotkaVolterraSimulator(
        num_timepoints=8,
        days=3.0,
        observation_noise_scale=0.1,
    )
    theta = np.array([0.8, 0.05, 0.8, 0.05])
    states = simulator._solve_states(theta).T.reshape(-1)

    actual = lotka_volterra_log_likelihood(np.log(theta), states, simulator)

    assert np.isfinite(actual)


def test_lotka_volterra_mcar_all_observed_recovers_full_likelihood() -> None:
    simulator = LotkaVolterraSimulator(num_timepoints=8, days=3.0)
    rng = np.random.default_rng(4)
    theta = np.array([0.8, 0.05, 0.8, 0.05])
    x = simulator.simulate(theta, rng)
    z = np.log(theta)
    np.testing.assert_allclose(
        lotka_volterra_log_likelihood(z, x, simulator),
        lotka_volterra_log_likelihood(z, x, simulator, mask=np.ones_like(x, dtype=np.int8)),
        rtol=0.0, atol=0.0,
    )


def test_lotka_volterra_mcar_masked_likelihood_matches_manual_slice() -> None:
    simulator = LotkaVolterraSimulator(num_timepoints=8, days=3.0)
    theta = np.array([0.8, 0.05, 0.8, 0.05])
    states = simulator._solve_states(theta).T.reshape(-1)
    x = states * np.exp(np.linspace(-0.08, 0.08, states.size))
    mask = np.tile(np.array([1, 1, 0, 0], dtype=np.int8), 4)
    actual = lotka_volterra_log_likelihood(np.log(theta), x, simulator, mask=mask)
    expected = np.sum(
        norm.logpdf(np.log(x[mask.astype(bool)]),
                    loc=np.log(np.clip(states[mask.astype(bool)], 1e-10, simulator.max_state)),
                    scale=simulator.observation_noise_scale)
        - np.log(x[mask.astype(bool)])
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-12)


def test_lotka_volterra_mcar_all_missing_recovers_log_prior_and_allows_zeros() -> None:
    simulator = LotkaVolterraSimulator(num_timepoints=8, days=3.0)
    z = np.array([-0.2, -2.9, -0.1, -3.1])
    x_obs = np.zeros(simulator.x_shape)
    mask = np.zeros(simulator.x_shape, dtype=np.int8)
    prior = np.sum(norm.logpdf(z, simulator.prior_log_mean, simulator.prior_log_std))
    assert lotka_volterra_log_likelihood(z, x_obs, simulator, mask=mask) == 0.0
    np.testing.assert_allclose(lotka_volterra_log_prob(z, x_obs, simulator, mask=mask), prior)


def test_lotka_volterra_mcar_time_blocks_mask_both_species_and_t0_is_not_measurement() -> None:
    simulator = LotkaVolterraSimulator(num_timepoints=8, days=3.0)
    theta = np.array([0.8, 0.05, 0.8, 0.05])
    x = simulator._solve_states(theta).T.reshape(-1)
    mask = LotkaVolterraTimeBlockMCARMask(0.25, block_size=2).generate(
        x, theta, np.random.default_rng(7)
    )
    assert np.all(mask.reshape(-1, 2)[:, 0] == mask.reshape(-1, 2)[:, 1])
    mask[:2] = 0
    x_obs = x * mask
    value = lotka_volterra_log_likelihood(np.log(theta), x_obs, simulator, mask=mask)
    assert np.isfinite(value)


def test_lotka_volterra_mcar_failed_ode_returns_negative_infinity(monkeypatch: pytest.MonkeyPatch) -> None:
    simulator = LotkaVolterraSimulator(num_timepoints=8, days=3.0)
    monkeypatch.setattr(simulator, "_solve_states", lambda _theta: (_ for _ in ()).throw(FloatingPointError()))
    x = np.ones(simulator.x_shape)
    assert lotka_volterra_log_likelihood(np.zeros(4), x, simulator, mask=np.ones(simulator.x_shape)) == -np.inf


def test_lotka_volterra_log_total_mnar_log_probability_matches_generator() -> None:
    x = np.tile(np.array([1.0, 2.0]), 8)
    generator = LotkaVolterraLogTotalMNARMask(.5)
    mask = generator.generate(x, None, np.random.default_rng(7))
    lp = lotka_volterra_log_total_mnar_log_probability(x, mask, missing_fraction=.5)
    assert np.isfinite(lp)
    broken = mask.copy(); broken[1] = 1 - broken[1]
    assert lotka_volterra_log_total_mnar_log_probability(x, broken, missing_fraction=.5) == -np.inf


def test_lotka_volterra_log_total_mnar_probability_matches_empirical_frequency() -> None:
    # Two time points yield p_missing=(0, 0.5) at missing_fraction=0.25.
    x = np.array([1.0, 1.0, 3.0, 3.0])
    generator = LotkaVolterraLogTotalMNARMask(.25)
    target = np.array([1, 1, 0, 0], dtype=np.int8)
    probability = np.exp(lotka_volterra_log_total_mnar_log_probability(x, target, missing_fraction=.25))
    rng = np.random.default_rng(3)
    empirical = np.mean([np.array_equal(generator.generate(x, None, rng), target) for _ in range(10_000)])
    assert abs(empirical - probability) < .025


def test_lotka_volterra_soft_mnar_likelihood_has_correct_limits() -> None:
    x = np.array([1.0, 1.0, 3.0, 3.0])
    impossible = np.array([0, 0, 1, 1], dtype=np.int8)
    assert lotka_volterra_log_total_mnar_log_probability(x, impossible, missing_fraction=.25) == -np.inf
    assert np.isfinite(lotka_volterra_log_total_mnar_soft_log_probability(x, impossible, missing_fraction=.25, delta=.05))
    target = np.array([1, 1, 0, 0], dtype=np.int8)
    exact = lotka_volterra_log_total_mnar_log_probability(x, target, missing_fraction=.25)
    assert lotka_volterra_log_total_mnar_soft_log_probability(x, target, missing_fraction=.25, delta=0.0) == exact


def test_partial_reference_hdf5_preserves_sparse_original_indices(tmp_path: Path) -> None:
    path = tmp_path / "sparse.h5"
    observations = {"theta_true": np.ones((2, 1)), "x_full": np.ones((2, 2)),
                    "x_obs": np.ones((2, 2)), "mask": np.ones((2, 2), dtype=np.int8),
                    "reference_indices": np.array([0, 7], dtype=np.int64)}
    save_partial_data_reference_posteriors_hdf5(
        path, observations=observations, theta_samples=np.ones((2, 3, 1)), metadata={"problem": "test"}
    )
    loaded, _, _, _ = load_partial_data_reference_posteriors_hdf5(path)
    np.testing.assert_array_equal(loaded["reference_indices"], [0, 7])


def test_lotka_volterra_mnar_missing_sampler_keeps_observed_entries() -> None:
    simulator = LotkaVolterraSimulator(num_timepoints=8, days=3.0)
    x_obs = np.zeros(16); mask = np.tile(np.array([1, 1, 0, 0]), 4)
    x_obs[mask.astype(bool)] = 2.0
    draw = sample_lotka_volterra_mnar_missing_observations(np.log([.8,.05,.8,.05]), x_obs, mask, simulator=simulator, rng=np.random.default_rng(1))
    assert np.array_equal(draw[mask.astype(bool)], x_obs[mask.astype(bool)])
    assert np.all(draw[~mask.astype(bool)] > 0)


def test_lotka_volterra_screened_initialization_is_finite_local_and_nondegenerate() -> None:
    simulator = LotkaVolterraSimulator(num_timepoints=8, days=3.0)
    theta = np.array([0.8, 0.05, 0.8, 0.05])
    x = simulator._solve_states(theta).T.reshape(-1)
    mask = np.tile(np.array([1, 1, 0, 0], dtype=np.int8), 4)
    x_obs = x * mask
    map_z, map_lp = find_lotka_volterra_map(x_obs, simulator, mask=mask)
    walkers, diagnostics = initialize_screened_lotka_volterra_walkers(
        map_log_theta=map_z, map_log_prob=map_lp, simulator=simulator,
        x_full=x_obs, mask=mask, num_walkers=16, fallback_scale=0.3,
        rng=np.random.default_rng(21),
    )
    initial_lp = lotka_volterra_log_prob(walkers, x_obs, simulator, mask=mask)
    assert np.all(np.isfinite(initial_lp))
    assert np.all(initial_lp >= map_lp - diagnostics["screen_log_density_drop"])
    assert np.linalg.matrix_rank(walkers - walkers.mean(axis=0)) == simulator.theta_dim
    assert diagnostics["candidate_count"] >= 16
    assert diagnostics["rejection_count"] >= 0


def test_generate_lotka_volterra_reference_posteriors_contract_short_mcmc() -> None:
    observations, metadata, reference_data = generate_lotka_volterra_reference_posteriors(
        num_observations=1,
        num_reference_samples=20,
        observation_seed=123,
        posterior_seed=456,
        num_timepoints=8,
        days=3.0,
        num_walkers=12,
        burn_in_steps=8,
        production_steps=10,
        trace_num_steps=5,
        trace_num_walkers=3,
        validation_num_ensembles=0,
        validation_observation_indices=(),
    )

    theta_samples = reference_data["theta_samples"]
    assert observations["theta_true"].shape == (1, 4)
    assert observations["x_full"].shape == (1, 16)
    assert theta_samples.shape == (1, 20, 4)
    assert metadata["problem"] == "lotka_volterra"
    assert metadata["posterior_method"] == "emcee_log_space_exact_likelihood"
    assert metadata["sampling_space"] == "log_theta"
    assert reference_data["acceptance_fraction"].shape == (1,)
    assert reference_data["log_theta_split_rhat"].shape == (1, 4)
    assert reference_data["mcmc_trace"].shape == (1, 5, 3, 4)
    assert reference_data["log_theta_mcmc_trace"].shape == (1, 5, 3, 4)
    assert reference_data["num_nonpositive_theta_samples"] == 0
    assert np.all(theta_samples > 0)


def test_plot_reference_pair_saves_for_four_dimensional_references(tmp_path: Path) -> None:
    observations, metadata, reference_data = generate_lotka_volterra_reference_posteriors(
        num_observations=1,
        num_reference_samples=20,
        observation_seed=123,
        posterior_seed=456,
        num_timepoints=8,
        days=3.0,
        num_walkers=12,
        burn_in_steps=8,
        production_steps=10,
        trace_num_steps=5,
        trace_num_walkers=3,
        validation_num_ensembles=0,
        validation_observation_indices=(),
    )
    theta_samples = reference_data.pop("theta_samples")
    reference_path = tmp_path / "lotka_volterra_references.h5"
    save_reference_posteriors_hdf5(
        reference_path,
        observations=observations,
        theta_samples=theta_samples,
        metadata=metadata,
        diagnostics=reference_data,
    )

    paths = plot_reference_posterior_pairs(reference_path, max_samples=20)

    assert len(paths) == 1
    assert paths[0].exists()


def test_plot_lotka_volterra_reference_predictives_saves_file(tmp_path: Path) -> None:
    observations, metadata, reference_data = generate_lotka_volterra_reference_posteriors(
        num_observations=1,
        num_reference_samples=20,
        observation_seed=123,
        posterior_seed=456,
        num_timepoints=8,
        days=3.0,
        num_walkers=12,
        burn_in_steps=8,
        production_steps=10,
        trace_num_steps=5,
        trace_num_walkers=3,
        validation_num_ensembles=0,
        validation_observation_indices=(),
    )
    theta_samples = reference_data.pop("theta_samples")
    reference_path = tmp_path / "lotka_volterra_references.h5"
    save_reference_posteriors_hdf5(
        reference_path,
        observations=observations,
        theta_samples=theta_samples,
        metadata=metadata,
        diagnostics=reference_data,
    )

    paths = plot_lotka_volterra_reference_predictives(reference_path, max_samples=20)

    assert len(paths) == 1
    assert paths[0].exists()
