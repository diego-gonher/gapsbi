from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gapsbi.references import (
    REFERENCE_SCHEMA_NAME,
    compute_oup_grid_posterior,
    generate_glm_reference_posteriors,
    generate_glu_reference_posteriors,
    generate_lotka_volterra_reference_posteriors,
    generate_oup_reference_posteriors,
    glm_log_likelihood,
    load_reference_posteriors_hdf5,
    lotka_volterra_log_likelihood,
    plot_lotka_volterra_reference_predictives,
    oup_log_likelihood,
    plot_reference_mcmc_traces,
    plot_reference_posterior_marginals,
    plot_reference_posterior_pairs,
    save_reference_posteriors_hdf5,
    sample_glu_reference_posterior,
)
from gapsbi.simulators import GLMSimulator, LotkaVolterraSimulator, OUPSimulator


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
    assert reference_data["num_prior_bound_violations"] == 0
    assert np.all(theta_samples >= -2.0)
    assert np.all(theta_samples <= 2.0)


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
