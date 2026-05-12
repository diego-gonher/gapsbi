import numpy as np
import pytest

from gapsbi.simulators import GLMSimulator, GLUSimulator, OUPSimulator, RickerSimulator


def test_ricker_sample_theta_shape() -> None:
    simulator = RickerSimulator()
    theta = simulator.sample_theta(7, np.random.default_rng(123))

    assert theta.shape == (7, 2)


def test_ricker_sample_theta_within_prior_bounds() -> None:
    simulator = RickerSimulator()
    theta = simulator.sample_theta(100, np.random.default_rng(123))

    assert np.all(theta >= simulator.prior_low)
    assert np.all(theta <= simulator.prior_high)


def test_ricker_simulate_single_theta_shape() -> None:
    simulator = RickerSimulator(T=12)
    theta = np.array([4.0, 10.0])
    x = simulator.simulate(theta, np.random.default_rng(123))

    assert x.shape == (12,)


def test_ricker_simulate_batch_theta_shape() -> None:
    simulator = RickerSimulator(T=12)
    theta = np.array([[4.0, 10.0], [5.0, 2.0], [3.0, 0.0]])
    x = simulator.simulate(theta, np.random.default_rng(123))

    assert x.shape == (3, 12)


def test_ricker_same_seed_produces_identical_samples() -> None:
    simulator_a = RickerSimulator(T=12)
    simulator_b = RickerSimulator(T=12)
    theta = np.array([[4.0, 10.0], [5.0, 2.0]])
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    theta_a = simulator_a.sample_theta(5, rng_a)
    theta_b = simulator_b.sample_theta(5, rng_b)
    x_a = simulator_a.simulate(theta, rng_a)
    x_b = simulator_b.simulate(theta, rng_b)

    np.testing.assert_array_equal(theta_a, theta_b)
    np.testing.assert_array_equal(x_a, x_b)


def test_ricker_simulate_raises_for_negative_phi() -> None:
    simulator = RickerSimulator()

    with pytest.raises(ValueError, match="phi must be nonnegative"):
        simulator.simulate(np.array([4.0, -1.0]), np.random.default_rng(123))


def test_ricker_output_is_finite_and_nonnegative() -> None:
    simulator = RickerSimulator(T=25)
    theta = np.array([[8.0, 20.0], [2.0, 0.0], [6.0, 5.0]])
    x = simulator.simulate(theta, np.random.default_rng(123))

    assert np.all(np.isfinite(x))
    assert np.all(x >= 0)


def test_oup_sample_theta_shape() -> None:
    simulator = OUPSimulator()
    theta = simulator.sample_theta(7, np.random.default_rng(123))

    assert theta.shape == (7, 2)


def test_oup_sample_theta_within_prior_bounds() -> None:
    simulator = OUPSimulator()
    theta = simulator.sample_theta(100, np.random.default_rng(123))

    assert np.all(theta >= simulator.prior_low)
    assert np.all(theta <= simulator.prior_high)


def test_oup_simulate_single_theta_shape() -> None:
    simulator = OUPSimulator(n=12)
    theta = np.array([1.0, 0.0])
    x = simulator.simulate(theta, np.random.default_rng(123))

    assert x.shape == (12,)


def test_oup_simulate_batch_theta_shape() -> None:
    simulator = OUPSimulator(n=12)
    theta = np.array([[1.0, 0.0], [0.5, 1.0], [2.0, -1.0]])
    x = simulator.simulate(theta, np.random.default_rng(123))

    assert x.shape == (3, 12)


def test_oup_same_seed_produces_identical_samples() -> None:
    simulator_a = OUPSimulator(n=12)
    simulator_b = OUPSimulator(n=12)
    theta = np.array([[1.0, 0.0], [0.5, 1.0]])
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    theta_a = simulator_a.sample_theta(5, rng_a)
    theta_b = simulator_b.sample_theta(5, rng_b)
    x_a = simulator_a.simulate(theta, rng_a)
    x_b = simulator_b.simulate(theta, rng_b)

    np.testing.assert_array_equal(theta_a, theta_b)
    np.testing.assert_array_equal(x_a, x_b)


def test_oup_invalid_theta_shape_raises() -> None:
    simulator = OUPSimulator()

    with pytest.raises(ValueError, match="theta must have shape"):
        simulator.simulate(np.array([1.0, 0.0, 2.0]), np.random.default_rng(123))

    with pytest.raises(ValueError, match="theta must have shape"):
        simulator.simulate(np.ones((2, 3)), np.random.default_rng(123))


def test_oup_negative_theta1_raises() -> None:
    simulator = OUPSimulator()

    with pytest.raises(ValueError, match="theta1 must be nonnegative"):
        simulator.simulate(np.array([-1.0, 0.0]), np.random.default_rng(123))


def test_oup_output_is_finite() -> None:
    simulator = OUPSimulator(n=25)
    theta = np.array([[0.0, -2.0], [1.0, 0.0], [2.0, 2.0]])
    x = simulator.simulate(theta, np.random.default_rng(123))

    assert np.all(np.isfinite(x))


def test_glu_sample_theta_shape() -> None:
    simulator = GLUSimulator(dim=6)
    theta = simulator.sample_theta(7, np.random.default_rng(123))

    assert theta.shape == (7, 6)


def test_glu_sample_theta_within_prior_bounds() -> None:
    simulator = GLUSimulator(dim=6, prior_bound=2.0)
    theta = simulator.sample_theta(100, np.random.default_rng(123))

    assert np.all(theta >= -2.0)
    assert np.all(theta <= 2.0)


def test_glu_simulate_single_theta_shape() -> None:
    simulator = GLUSimulator(dim=6)
    theta = np.zeros(6)
    x = simulator.simulate(theta, np.random.default_rng(123))

    assert x.shape == (6,)


def test_glu_simulate_batch_theta_shape() -> None:
    simulator = GLUSimulator(dim=6)
    theta = np.zeros((3, 6))
    x = simulator.simulate(theta, np.random.default_rng(123))

    assert x.shape == (3, 6)


def test_glu_same_seed_produces_identical_samples() -> None:
    simulator_a = GLUSimulator(dim=6)
    simulator_b = GLUSimulator(dim=6)
    theta = np.zeros((2, 6))
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    theta_a = simulator_a.sample_theta(5, rng_a)
    theta_b = simulator_b.sample_theta(5, rng_b)
    x_a = simulator_a.simulate(theta, rng_a)
    x_b = simulator_b.simulate(theta, rng_b)

    np.testing.assert_array_equal(theta_a, theta_b)
    np.testing.assert_array_equal(x_a, x_b)


def test_glu_invalid_theta_shape_raises() -> None:
    simulator = GLUSimulator(dim=6)

    with pytest.raises(ValueError, match="theta must have shape"):
        simulator.simulate(np.zeros(5), np.random.default_rng(123))

    with pytest.raises(ValueError, match="theta must have shape"):
        simulator.simulate(np.zeros((2, 5)), np.random.default_rng(123))


def test_glu_output_is_finite() -> None:
    simulator = GLUSimulator(dim=6)
    theta = np.zeros((3, 6))
    x = simulator.simulate(theta, np.random.default_rng(123))

    assert np.all(np.isfinite(x))


def test_glu_zero_simulator_scale_returns_theta_exactly() -> None:
    simulator = GLUSimulator(dim=6, simulator_scale=0.0)
    theta = np.arange(6, dtype=float)
    x = simulator.simulate(theta, np.random.default_rng(123))

    np.testing.assert_array_equal(x, theta)


def test_glm_sample_theta_shape() -> None:
    simulator = GLMSimulator(dim=6)
    theta = simulator.sample_theta(7, np.random.default_rng(123))

    assert theta.shape == (7, 6)


def test_glm_sample_theta_within_prior_bounds() -> None:
    simulator = GLMSimulator(dim=6, prior_bound=3.0)
    theta = simulator.sample_theta(100, np.random.default_rng(123))

    assert np.all(theta >= -3.0)
    assert np.all(theta <= 3.0)


def test_glm_simulate_single_theta_shape_for_summary_mode() -> None:
    simulator = GLMSimulator(dim=6, duration=20)
    theta = np.zeros(6)
    x = simulator.simulate(theta, np.random.default_rng(123))

    assert x.shape == (6,)


def test_glm_simulate_batch_theta_shape_for_summary_mode() -> None:
    simulator = GLMSimulator(dim=6, duration=20)
    theta = np.zeros((3, 6))
    x = simulator.simulate(theta, np.random.default_rng(123))

    assert x.shape == (3, 6)


def test_glm_raw_mode_returns_binary_output() -> None:
    simulator = GLMSimulator(dim=6, duration=20, summary="raw")
    theta = np.zeros(6)
    x = simulator.simulate(theta, np.random.default_rng(123))

    assert x.shape == (20,)
    assert np.all((x == 0) | (x == 1))


def test_glm_same_seed_produces_identical_samples() -> None:
    simulator_a = GLMSimulator(dim=6, duration=20)
    simulator_b = GLMSimulator(dim=6, duration=20)
    theta = np.zeros((2, 6))
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    theta_a = simulator_a.sample_theta(5, rng_a)
    theta_b = simulator_b.sample_theta(5, rng_b)
    x_a = simulator_a.simulate(theta, rng_a)
    x_b = simulator_b.simulate(theta, rng_b)

    np.testing.assert_array_equal(theta_a, theta_b)
    np.testing.assert_array_equal(x_a, x_b)


def test_glm_invalid_theta_shape_raises() -> None:
    simulator = GLMSimulator(dim=6)

    with pytest.raises(ValueError, match="theta must have shape"):
        simulator.simulate(np.zeros(5), np.random.default_rng(123))

    with pytest.raises(ValueError, match="theta must have shape"):
        simulator.simulate(np.zeros((2, 5)), np.random.default_rng(123))


def test_glm_output_is_finite() -> None:
    simulator = GLMSimulator(dim=6, duration=20)
    theta = np.zeros((3, 6))
    x = simulator.simulate(theta, np.random.default_rng(123))

    assert np.all(np.isfinite(x))


def test_glm_metadata_contains_stimulus_seed_and_summary() -> None:
    simulator = GLMSimulator(dim=6, duration=20, stimulus_seed=99, summary="raw")
    metadata = simulator.metadata()

    assert metadata["stimulus_seed"] == 99
    assert metadata["summary"] == "raw"
