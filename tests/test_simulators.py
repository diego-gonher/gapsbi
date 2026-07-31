import numpy as np
import pytest

from gapsbi.datasets import generate_dataset, generate_split
from gapsbi.masks import (
    BlockMCARMask,
    CoordinateMARMask,
    LotkaVolterraLogTotalMNARMask,
    LotkaVolterraTimeBlockMCARMask,
    LotkaVolterraTimeMARMask,
    PointMCARMask,
    SelfCensoringMNARMask,
)
from gapsbi.rng import make_rng
from gapsbi.simulators import (
    GLMSimulator,
    GLUSimulator,
    HodgkinHuxleySimulator,
    LotkaVolterraSimulator,
    OUPSimulator,
    RickerSimulator,
    SpatialSIRSimulator,
)


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


def test_spatial_sir_basic_properties() -> None:
    simulator = SpatialSIRSimulator(lattice_shape=(4, 5))

    assert simulator.theta_dim == 2
    assert simulator.x_shape == (3 * 4 * 5,)


def test_spatial_sir_simulate_single_shape_dtype_and_values() -> None:
    simulator = SpatialSIRSimulator(
        lattice_shape=(4, 4),
        measurement_time=0.05,
        simulation_step_size=0.01,
    )
    x = simulator.simulate(np.array([0.5, 0.2]), np.random.default_rng(123))

    assert x.shape == (3 * 4 * 4,)
    assert x.dtype == np.float32
    assert np.all((x == 0.0) | (x == 1.0))


def test_spatial_sir_simulate_batch_shape() -> None:
    simulator = SpatialSIRSimulator(
        lattice_shape=(4, 4),
        measurement_time=0.05,
        simulation_step_size=0.01,
    )
    theta = np.array([[0.5, 0.2], [0.1, 0.8]])
    x = simulator.simulate(theta, np.random.default_rng(123))

    assert x.shape == (2, 3 * 4 * 4)


def test_spatial_sir_same_seed_is_reproducible() -> None:
    simulator = SpatialSIRSimulator(
        lattice_shape=(4, 4),
        measurement_time=0.05,
        simulation_step_size=0.01,
    )
    theta = np.array([[0.5, 0.2], [0.1, 0.8]])
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    theta_a = simulator.sample_theta(3, rng_a)
    theta_b = simulator.sample_theta(3, rng_b)
    x_a = simulator.simulate(theta, rng_a)
    x_b = simulator.simulate(theta, rng_b)

    np.testing.assert_array_equal(theta_a, theta_b)
    np.testing.assert_array_equal(x_a, x_b)


def test_spatial_sir_different_theta_produces_valid_outputs() -> None:
    simulator = SpatialSIRSimulator(
        lattice_shape=(4, 4),
        measurement_time=0.05,
        simulation_step_size=0.01,
    )
    theta = np.array([[0.0, 0.0], [1.0, 1.0]])
    x = simulator.simulate(theta, np.random.default_rng(123))

    assert x.shape == (2, 3 * 4 * 4)
    assert np.all((x == 0.0) | (x == 1.0))


def test_spatial_sir_metadata_includes_original_x_shape() -> None:
    simulator = SpatialSIRSimulator(lattice_shape=(4, 5))
    metadata = simulator.metadata()

    assert metadata["original_x_shape"] == (3, 4, 5)


def test_spatial_sir_is_compatible_with_generate_split_and_dataset() -> None:
    simulator = SpatialSIRSimulator(
        lattice_shape=(4, 4),
        measurement_time=0.05,
        simulation_step_size=0.01,
    )
    mask = PointMCARMask(missing_fraction=0.25)

    split = generate_split(simulator, mask, n=3, rng=make_rng(123))
    dataset = generate_dataset(simulator, mask, n_train=3, n_val=2, n_test=1, seed=123)

    assert split["x_full"].shape == (3, 3 * 4 * 4)
    assert dataset["train"]["x_full"].shape == (3, 3 * 4 * 4)


@pytest.mark.parametrize(
    "mask_generator",
    [
        PointMCARMask(missing_fraction=0.25),
        BlockMCARMask(missing_fraction=0.25, block_size=3),
        CoordinateMARMask(missing_fraction=0.25),
        SelfCensoringMNARMask(missing_fraction=0.25),
    ],
)
def test_spatial_sir_spatial_masks_share_status_across_channels(mask_generator) -> None:
    simulator = SpatialSIRSimulator(
        lattice_shape=(4, 4),
        measurement_time=0.05,
        simulation_step_size=0.01,
    )
    split = generate_split(simulator, mask_generator, n=2, rng=make_rng(123))

    mask = split["mask"].reshape((2, 3, 4, 4))

    assert split["mask"].shape == split["x_full"].shape
    assert np.all(mask[:, 0] == mask[:, 1])
    assert np.all(mask[:, 1] == mask[:, 2])


def test_hodgkin_huxley_basic_properties() -> None:
    simulator = HodgkinHuxleySimulator()

    assert simulator.theta_dim == 2
    assert simulator.x_shape == (601,)


def test_hodgkin_huxley_simulate_single_shape_dtype_and_finite() -> None:
    simulator = HodgkinHuxleySimulator(
        duration=1.0,
        dt=0.05,
        t_on=0.2,
        downsample=2,
    )
    x = simulator.simulate(np.array([20.0, 5.0]), np.random.default_rng(123))

    assert x.shape == simulator.x_shape
    assert x.dtype == np.float32
    assert np.all(np.isfinite(x))


def test_hodgkin_huxley_simulate_batch_shape() -> None:
    simulator = HodgkinHuxleySimulator(
        duration=1.0,
        dt=0.05,
        t_on=0.2,
        downsample=2,
    )
    theta = np.array([[20.0, 5.0], [40.0, 8.0]])
    x = simulator.simulate(theta, np.random.default_rng(123))

    assert x.shape == (2, *simulator.x_shape)


def test_hodgkin_huxley_same_seed_is_reproducible() -> None:
    simulator = HodgkinHuxleySimulator(
        duration=1.0,
        dt=0.05,
        t_on=0.2,
        downsample=2,
    )
    theta = np.array([[20.0, 5.0], [40.0, 8.0]])
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    theta_a = simulator.sample_theta(3, rng_a)
    theta_b = simulator.sample_theta(3, rng_b)
    x_a = simulator.simulate(theta, rng_a)
    x_b = simulator.simulate(theta, rng_b)

    np.testing.assert_array_equal(theta_a, theta_b)
    np.testing.assert_array_equal(x_a, x_b)


def test_hodgkin_huxley_different_theta_produces_valid_outputs() -> None:
    simulator = HodgkinHuxleySimulator(
        duration=1.0,
        dt=0.05,
        t_on=0.2,
        downsample=2,
    )
    theta = np.array([[0.5, 1e-4], [80.0, 15.0]])
    x = simulator.simulate(theta, np.random.default_rng(123))

    assert x.shape == (2, *simulator.x_shape)
    assert np.all(np.isfinite(x))


def test_hodgkin_huxley_metadata_includes_downsample_and_full_trace_length() -> None:
    simulator = HodgkinHuxleySimulator()
    metadata = simulator.metadata()

    assert metadata["downsample"] == 20
    assert metadata["full_trace_length"] == 12001


def test_hodgkin_huxley_is_compatible_with_generate_split_and_dataset() -> None:
    simulator = HodgkinHuxleySimulator(
        duration=1.0,
        dt=0.05,
        t_on=0.2,
        downsample=2,
    )
    mask = PointMCARMask(missing_fraction=0.25)

    split = generate_split(simulator, mask, n=3, rng=make_rng(123))
    dataset = generate_dataset(simulator, mask, n_train=3, n_val=2, n_test=1, seed=123)

    assert split["x_full"].shape == (3, *simulator.x_shape)
    assert dataset["train"]["x_full"].shape == (3, *simulator.x_shape)


@pytest.mark.parametrize(
    "mask_generator",
    [
        PointMCARMask(missing_fraction=0.25),
        BlockMCARMask(missing_fraction=0.25, block_size=3),
        CoordinateMARMask(missing_fraction=0.25),
        SelfCensoringMNARMask(missing_fraction=0.25),
    ],
)
def test_hodgkin_huxley_is_compatible_with_current_masks(mask_generator) -> None:
    simulator = HodgkinHuxleySimulator(
        duration=1.0,
        dt=0.05,
        t_on=0.2,
        downsample=2,
    )
    split = generate_split(simulator, mask_generator, n=2, rng=make_rng(123))

    assert split["mask"].shape == split["x_full"].shape


def test_lotka_volterra_basic_properties() -> None:
    simulator = LotkaVolterraSimulator(num_timepoints=50)

    assert simulator.theta_dim == 4
    assert simulator.x_shape == (100,)


def test_lotka_volterra_sample_theta_is_positive_lognormal() -> None:
    simulator = LotkaVolterraSimulator(num_timepoints=12)
    theta = simulator.sample_theta(8, np.random.default_rng(123))

    assert theta.shape == (8, 4)
    assert np.all(theta > 0)


def test_lotka_volterra_simulate_single_shape_positive_and_finite() -> None:
    simulator = LotkaVolterraSimulator(num_timepoints=12, days=5.0)
    theta = np.array([0.8, 0.05, 0.8, 0.05])
    x = simulator.simulate(theta, np.random.default_rng(123))

    assert x.shape == (24,)
    assert np.all(np.isfinite(x))
    assert np.all(x > 0)


def test_lotka_volterra_simulate_batch_shape() -> None:
    simulator = LotkaVolterraSimulator(num_timepoints=12, days=5.0)
    theta = np.array([[0.8, 0.05, 0.8, 0.05], [0.5, 0.03, 1.0, 0.04]])
    x = simulator.simulate(theta, np.random.default_rng(123))

    assert x.shape == (2, 24)


def test_lotka_volterra_same_seed_is_reproducible() -> None:
    simulator_a = LotkaVolterraSimulator(num_timepoints=12, days=5.0)
    simulator_b = LotkaVolterraSimulator(num_timepoints=12, days=5.0)
    theta = np.array([[0.8, 0.05, 0.8, 0.05], [0.5, 0.03, 1.0, 0.04]])
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    theta_a = simulator_a.sample_theta(5, rng_a)
    theta_b = simulator_b.sample_theta(5, rng_b)
    x_a = simulator_a.simulate(theta, rng_a)
    x_b = simulator_b.simulate(theta, rng_b)

    np.testing.assert_array_equal(theta_a, theta_b)
    np.testing.assert_array_equal(x_a, x_b)


def test_lotka_volterra_metadata_documents_interleaved_layout_and_prior() -> None:
    simulator = LotkaVolterraSimulator(num_timepoints=12, days=5.0)
    metadata = simulator.metadata()

    assert metadata["name"] == "lotka_volterra"
    assert metadata["observation_layout"] == "interleaved_prey_predator"
    assert metadata["prior"] == "lognormal"
    assert len(metadata["timepoints"]) == 12


@pytest.mark.parametrize(
    "mask_generator",
    [
        PointMCARMask(missing_fraction=0.25),
        LotkaVolterraTimeBlockMCARMask(missing_fraction=0.25, block_size=3),
        LotkaVolterraTimeMARMask(missing_fraction=0.25),
        LotkaVolterraLogTotalMNARMask(missing_fraction=0.25),
    ],
)
def test_lotka_volterra_is_compatible_with_generate_split_and_dataset(mask_generator) -> None:
    simulator = LotkaVolterraSimulator(num_timepoints=12, days=5.0)

    split = generate_split(simulator, mask_generator, n=3, rng=make_rng(123))
    dataset = generate_dataset(simulator, mask_generator, n_train=3, n_val=2, n_test=1, seed=123)

    assert split["x_full"].shape == (3, 24)
    assert split["mask"].shape == split["x_full"].shape
    assert dataset["train"]["x_full"].shape == (3, 24)


@pytest.mark.parametrize(
    "mask_generator",
    [
        LotkaVolterraTimeBlockMCARMask(missing_fraction=0.25, block_size=3),
        LotkaVolterraTimeMARMask(missing_fraction=0.25),
        LotkaVolterraLogTotalMNARMask(missing_fraction=0.25),
    ],
)
def test_lotka_volterra_time_masks_share_status_across_populations(mask_generator) -> None:
    simulator = LotkaVolterraSimulator(num_timepoints=12, days=5.0)
    split = generate_split(simulator, mask_generator, n=4, rng=make_rng(123))
    mask = split["mask"].reshape((4, 12, 2))

    assert np.all(mask[:, :, 0] == mask[:, :, 1])
