import numpy as np
import pytest

from gapsbi.masks import (
    BlockMCARMask,
    CoordinateMARMask,
    LotkaVolterraLogTotalMNARMask,
    LotkaVolterraTimeBlockMCARMask,
    LotkaVolterraTimeMARMask,
    MeanNormalizedSelfCensoringMNARMask,
    PointMCARMask,
    SelfCensoringMNARMask,
)
from gapsbi.simulators import (
    GLMSimulator,
    GLUSimulator,
    LotkaVolterraSimulator,
    OUPSimulator,
    RickerSimulator,
)


def _assert_binary(mask: np.ndarray) -> None:
    assert np.all((mask == 0) | (mask == 1))


def test_point_mcar_mask_has_same_shape_and_is_binary() -> None:
    x_full = np.ones((5, 100))
    mask = PointMCARMask(missing_fraction=0.25).generate(
        x_full,
        theta=None,
        rng=np.random.default_rng(123),
    )

    assert mask.shape == x_full.shape
    _assert_binary(mask)


def test_point_mcar_same_seed_gives_same_mask() -> None:
    x_full = np.ones((5, 100))
    generator = PointMCARMask(missing_fraction=0.25)

    mask1 = generator.generate(x_full, None, np.random.default_rng(123))
    mask2 = generator.generate(x_full, None, np.random.default_rng(123))

    np.testing.assert_array_equal(mask1, mask2)


def test_point_mcar_different_seed_gives_different_mask() -> None:
    x_full = np.ones((5, 100))
    generator = PointMCARMask(missing_fraction=0.25)

    mask1 = generator.generate(x_full, None, np.random.default_rng(123))
    mask2 = generator.generate(x_full, None, np.random.default_rng(456))

    assert not np.array_equal(mask1, mask2)


def test_point_mcar_observed_fraction_is_approximately_expected() -> None:
    x_full = np.ones(10_000)
    missing_fraction = 0.25
    mask = PointMCARMask(missing_fraction=missing_fraction).generate(
        x_full,
        theta=None,
        rng=np.random.default_rng(123),
    )

    assert np.mean(mask) == pytest.approx(1.0 - missing_fraction, abs=0.03)


def test_block_mcar_mask_has_same_shape_and_is_binary() -> None:
    x_full = np.ones((4, 100))
    mask = BlockMCARMask(missing_fraction=0.25, block_size=5).generate(
        x_full,
        theta=None,
        rng=np.random.default_rng(123),
    )

    assert mask.shape == x_full.shape
    _assert_binary(mask)


def test_block_mcar_same_seed_gives_same_mask() -> None:
    x_full = np.ones((4, 100))
    generator = BlockMCARMask(missing_fraction=0.25, block_size=5)

    mask1 = generator.generate(x_full, None, np.random.default_rng(123))
    mask2 = generator.generate(x_full, None, np.random.default_rng(123))

    np.testing.assert_array_equal(mask1, mask2)


def test_block_mcar_different_seed_gives_different_mask() -> None:
    x_full = np.ones((4, 100))
    generator = BlockMCARMask(missing_fraction=0.25, block_size=5)

    mask1 = generator.generate(x_full, None, np.random.default_rng(123))
    mask2 = generator.generate(x_full, None, np.random.default_rng(456))

    assert not np.array_equal(mask1, mask2)


def test_block_mcar_observed_fraction_is_approximately_expected() -> None:
    x_full = np.ones((20, 200))
    missing_fraction = 0.25
    mask = BlockMCARMask(missing_fraction=missing_fraction, block_size=5).generate(
        x_full,
        theta=None,
        rng=np.random.default_rng(123),
    )

    assert np.mean(mask) == pytest.approx(1.0 - missing_fraction, abs=0.05)


def test_block_mcar_creates_contiguous_missing_regions_for_1d_data() -> None:
    x_full = np.ones(100)
    block_size = 5
    mask = BlockMCARMask(missing_fraction=0.2, block_size=block_size).generate(
        x_full,
        theta=None,
        rng=np.random.default_rng(123),
    )
    missing = np.flatnonzero(mask == 0)
    runs = np.split(missing, np.flatnonzero(np.diff(missing) > 1) + 1)

    assert any(run.size >= block_size for run in runs)


@pytest.mark.parametrize("mask_cls", [PointMCARMask, BlockMCARMask])
def test_invalid_missing_fraction_raises(mask_cls: type) -> None:
    with pytest.raises(ValueError, match="missing_fraction must be in"):
        if mask_cls is BlockMCARMask:
            mask_cls(missing_fraction=-0.1, block_size=5)
        else:
            mask_cls(missing_fraction=-0.1)

    with pytest.raises(ValueError, match="missing_fraction must be in"):
        if mask_cls is BlockMCARMask:
            mask_cls(missing_fraction=1.1, block_size=5)
        else:
            mask_cls(missing_fraction=1.1)


def test_invalid_block_size_raises() -> None:
    with pytest.raises(ValueError, match="block_size must be >= 1"):
        BlockMCARMask(missing_fraction=0.25, block_size=0)


def test_self_censoring_mnar_mask_preserves_unbatched_shape_and_is_binary() -> None:
    x_full = np.linspace(-2.0, 2.0, 25)
    mask = SelfCensoringMNARMask(missing_fraction=0.5).generate(
        x_full,
        theta=None,
        rng=np.random.default_rng(123),
    )

    assert mask.shape == x_full.shape
    _assert_binary(mask)


def test_self_censoring_mnar_mask_preserves_batched_shape_and_is_binary() -> None:
    x_full = np.tile(np.linspace(-2.0, 2.0, 25), (4, 1))
    mask = SelfCensoringMNARMask(missing_fraction=0.5).generate(
        x_full,
        theta=None,
        rng=np.random.default_rng(123),
    )

    assert mask.shape == x_full.shape
    _assert_binary(mask)


def test_self_censoring_mnar_same_seed_gives_same_mask() -> None:
    x_full = np.tile(np.linspace(-2.0, 2.0, 25), (4, 1))
    generator = SelfCensoringMNARMask(missing_fraction=0.5)

    mask1 = generator.generate(x_full, None, np.random.default_rng(123))
    mask2 = generator.generate(x_full, None, np.random.default_rng(123))

    np.testing.assert_array_equal(mask1, mask2)


def test_self_censoring_mnar_missingness_increases_with_normalized_value() -> None:
    x_full = np.tile(np.linspace(-1.0, 1.0, 5), (5_000, 1))
    mask = SelfCensoringMNARMask(missing_fraction=0.8).generate(
        x_full,
        theta=None,
        rng=np.random.default_rng(123),
    )
    missing_rate_by_value = np.mean(mask == 0, axis=0)

    assert missing_rate_by_value[-1] > missing_rate_by_value[0]
    assert missing_rate_by_value[-1] > missing_rate_by_value[2]


def test_self_censoring_mnar_works_with_negative_valued_data() -> None:
    x_full = np.array([[-3.0, -1.0, 0.0, 2.0], [-10.0, -5.0, -2.0, -1.0]])
    mask = SelfCensoringMNARMask(missing_fraction=0.5).generate(
        x_full,
        theta=None,
        rng=np.random.default_rng(123),
    )

    assert mask.shape == x_full.shape
    _assert_binary(mask)


def test_self_censoring_mnar_works_with_constant_data_without_nans() -> None:
    x_full = np.full((4, 10), 3.0)
    mask = SelfCensoringMNARMask(missing_fraction=0.5).generate(
        x_full,
        theta=None,
        rng=np.random.default_rng(123),
    )

    assert mask.shape == x_full.shape
    _assert_binary(mask)
    assert np.all(np.isfinite(mask))


@pytest.mark.parametrize(
    "simulator",
    [
        RickerSimulator(T=12),
        OUPSimulator(n=12),
        GLUSimulator(dim=6),
        GLMSimulator(dim=6, duration=20),
    ],
)
def test_self_censoring_mnar_is_compatible_with_implemented_simulators(simulator) -> None:
    rng = np.random.default_rng(123)
    theta = simulator.sample_theta(3, rng)
    x_full = simulator.simulate(theta, rng)
    mask = SelfCensoringMNARMask(missing_fraction=0.5).generate(x_full, theta, rng)

    assert mask.shape == x_full.shape
    _assert_binary(mask)


def test_self_censoring_mnar_invalid_missing_fraction_raises() -> None:
    with pytest.raises(ValueError, match="missing_fraction must be in"):
        SelfCensoringMNARMask(missing_fraction=-0.1)

    with pytest.raises(ValueError, match="missing_fraction must be in"):
        SelfCensoringMNARMask(missing_fraction=1.1)


def test_self_censoring_mnar_default_identity_matches_explicit_identity() -> None:
    x_full = np.tile(np.linspace(-2.0, 2.0, 25), (4, 1))

    default_mask = SelfCensoringMNARMask(missing_fraction=0.5).generate(
        x_full,
        theta=None,
        rng=np.random.default_rng(123),
    )
    explicit_mask = SelfCensoringMNARMask(
        missing_fraction=0.5,
        score_transform="identity",
    ).generate(
        x_full,
        theta=None,
        rng=np.random.default_rng(123),
    )

    np.testing.assert_array_equal(default_mask, explicit_mask)


def test_self_censoring_mnar_log1p_transform_works_for_nonnegative_data() -> None:
    x_full = np.array([[0.0, 1.0, 10.0, 100.0]])
    mask = SelfCensoringMNARMask(
        missing_fraction=1.0,
        score_transform="log1p",
    ).generate(
        x_full,
        theta=None,
        rng=np.random.default_rng(123),
    )

    assert mask.shape == x_full.shape
    _assert_binary(mask)


def test_self_censoring_mnar_log1p_transform_rejects_negative_data() -> None:
    x_full = np.array([-1.0, 0.0, 1.0])

    with pytest.raises(ValueError, match="log1p scoring requires nonnegative data"):
        SelfCensoringMNARMask(
            missing_fraction=0.5,
            score_transform="log1p",
        ).generate(
            x_full,
            theta=None,
            rng=np.random.default_rng(123),
        )


def test_self_censoring_mnar_abs_transform_works_for_signed_data() -> None:
    x_full = np.array([[-3.0, -1.0, 0.0, 2.0]])
    mask = SelfCensoringMNARMask(
        missing_fraction=0.5,
        score_transform="abs",
    ).generate(
        x_full,
        theta=None,
        rng=np.random.default_rng(123),
    )

    assert mask.shape == x_full.shape
    _assert_binary(mask)


def test_self_censoring_mnar_metadata_includes_score_transform() -> None:
    metadata = SelfCensoringMNARMask(
        missing_fraction=0.5,
        score_transform="log1p",
    ).metadata()

    assert metadata["score_transform"] == "log1p"


def test_self_censoring_mnar_invalid_score_transform_raises() -> None:
    with pytest.raises(ValueError, match="score_transform must be one of"):
        SelfCensoringMNARMask(missing_fraction=0.5, score_transform="sqrt")


def test_mean_normalized_self_censoring_mnar_realizes_epsilon_more_closely() -> None:
    x_full = np.tile(np.linspace(-1.0, 1.0, 10), (20_000, 1))
    mask = MeanNormalizedSelfCensoringMNARMask(missing_fraction=0.5).generate(
        x_full,
        theta=None,
        rng=np.random.default_rng(123),
    )
    missing_rate_by_value = np.mean(mask == 0, axis=0)

    assert np.mean(mask == 0) == pytest.approx(0.5, abs=0.02)
    assert missing_rate_by_value[-1] > missing_rate_by_value[0]
    assert missing_rate_by_value[-1] > missing_rate_by_value[2]


def test_mean_normalized_self_censoring_mnar_metadata_marks_variant() -> None:
    metadata = MeanNormalizedSelfCensoringMNARMask(
        missing_fraction=0.25,
        score_transform="identity",
    ).metadata()

    assert metadata["name"] == "self_censoring_mnar_mean_normalized"
    assert metadata["score"] == "mean_normalized_per_sample_minmax_shift"
    assert metadata["constant_score"] == 1.0


@pytest.mark.parametrize(
    "simulator",
    [
        OUPSimulator(n=12),
        GLUSimulator(dim=6),
        GLMSimulator(dim=6, duration=20),
    ],
)
def test_mean_normalized_self_censoring_mnar_is_compatible_with_main_generic_simulators(simulator) -> None:
    rng = np.random.default_rng(123)
    theta = simulator.sample_theta(3, rng)
    x_full = simulator.simulate(theta, rng)
    mask = MeanNormalizedSelfCensoringMNARMask(missing_fraction=0.5).generate(
        x_full,
        theta,
        rng,
    )

    assert mask.shape == x_full.shape
    _assert_binary(mask)


def test_coordinate_mar_mask_preserves_unbatched_shape_and_is_binary_int8() -> None:
    x_full = np.ones(25)
    mask = CoordinateMARMask(missing_fraction=0.25).generate(
        x_full,
        theta=None,
        rng=np.random.default_rng(123),
    )

    assert mask.shape == x_full.shape
    assert mask.dtype == np.int8
    _assert_binary(mask)


def test_coordinate_mar_mask_preserves_batched_shape_and_is_binary_int8() -> None:
    x_full = np.ones((4, 25))
    mask = CoordinateMARMask(missing_fraction=0.25).generate(
        x_full,
        theta=None,
        rng=np.random.default_rng(123),
    )

    assert mask.shape == x_full.shape
    assert mask.dtype == np.int8
    _assert_binary(mask)


def test_coordinate_mar_same_seed_gives_same_mask() -> None:
    x_full = np.ones((4, 25))
    generator = CoordinateMARMask(missing_fraction=0.25)

    mask1 = generator.generate(x_full, None, np.random.default_rng(123))
    mask2 = generator.generate(x_full, None, np.random.default_rng(123))

    np.testing.assert_array_equal(mask1, mask2)


def test_coordinate_mar_increasing_mode_masks_later_coordinates_more() -> None:
    x_full = np.ones((5_000, 20))
    mask = CoordinateMARMask(missing_fraction=0.4, mode="increasing").generate(
        x_full,
        theta=None,
        rng=np.random.default_rng(123),
    )
    missing_rate_by_coord = np.mean(mask == 0, axis=0)

    assert missing_rate_by_coord[-1] > missing_rate_by_coord[0]
    assert missing_rate_by_coord[-1] > missing_rate_by_coord[5]


def test_coordinate_mar_decreasing_mode_masks_earlier_coordinates_more() -> None:
    x_full = np.ones((5_000, 20))
    mask = CoordinateMARMask(missing_fraction=0.4, mode="decreasing").generate(
        x_full,
        theta=None,
        rng=np.random.default_rng(123),
    )
    missing_rate_by_coord = np.mean(mask == 0, axis=0)

    assert missing_rate_by_coord[0] > missing_rate_by_coord[-1]
    assert missing_rate_by_coord[0] > missing_rate_by_coord[-6]


def test_coordinate_mar_middle_mode_masks_middle_coordinates_more() -> None:
    x_full = np.ones((5_000, 21))
    mask = CoordinateMARMask(
        missing_fraction=0.4,
        mode="middle",
        middle_width=0.15,
    ).generate(
        x_full,
        theta=None,
        rng=np.random.default_rng(123),
    )
    missing_rate_by_coord = np.mean(mask == 0, axis=0)

    assert missing_rate_by_coord[10] > missing_rate_by_coord[0]
    assert missing_rate_by_coord[10] > missing_rate_by_coord[-1]


def test_coordinate_mar_does_not_depend_on_x_values() -> None:
    x_a = np.zeros((4, 25))
    x_b = np.arange(100, dtype=float).reshape(4, 25) * 1000.0
    generator = CoordinateMARMask(missing_fraction=0.25, mode="increasing")

    mask_a = generator.generate(x_a, None, np.random.default_rng(123))
    mask_b = generator.generate(x_b, None, np.random.default_rng(123))

    np.testing.assert_array_equal(mask_a, mask_b)


def test_coordinate_mar_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        CoordinateMARMask(missing_fraction=0.25, mode="late")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"missing_fraction": -0.1}, "missing_fraction must be in"),
        ({"missing_fraction": 1.1}, "missing_fraction must be in"),
        ({"missing_fraction": 0.25, "floor": -0.1}, "floor must be >= 0"),
        (
            {"missing_fraction": 0.25, "max_probability": -0.1},
            "max_probability must be in",
        ),
        (
            {"missing_fraction": 0.25, "max_probability": 1.1},
            "max_probability must be in",
        ),
        (
            {"missing_fraction": 0.25, "middle_width": 0.0},
            "middle_width must be > 0",
        ),
    ],
)
def test_coordinate_mar_invalid_parameters_raise(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        CoordinateMARMask(**kwargs)


@pytest.mark.parametrize(
    "simulator",
    [
        RickerSimulator(T=12),
        OUPSimulator(n=12),
        GLUSimulator(dim=6),
        GLMSimulator(dim=6, duration=20),
    ],
)
def test_coordinate_mar_is_compatible_with_implemented_simulators(simulator) -> None:
    rng = np.random.default_rng(123)
    theta = simulator.sample_theta(3, rng)
    x_full = simulator.simulate(theta, rng)
    mask = CoordinateMARMask(missing_fraction=0.25).generate(x_full, theta, rng)

    assert mask.shape == x_full.shape
    assert mask.dtype == np.int8
    _assert_binary(mask)


def test_lotka_volterra_time_block_mcar_pairs_population_entries() -> None:
    x_full = np.ones((4, 100))
    mask = LotkaVolterraTimeBlockMCARMask(
        missing_fraction=0.25,
        block_size=5,
    ).generate(x_full, theta=None, rng=np.random.default_rng(123))
    reshaped = mask.reshape((4, 50, 2))

    assert mask.shape == x_full.shape
    assert mask.dtype == np.int8
    _assert_binary(mask)
    assert np.all(reshaped[:, :, 0] == reshaped[:, :, 1])
    assert np.all(np.sum(reshaped[:, :, 0] == 0, axis=1) == round(0.25 * 50))


def test_lotka_volterra_time_mar_increasing_masks_later_times_more() -> None:
    x_full = np.ones((5_000, 100))
    mask = LotkaVolterraTimeMARMask(
        missing_fraction=0.4,
        mode="increasing",
    ).generate(x_full, theta=None, rng=np.random.default_rng(123))
    missing_rate_by_time = np.mean(mask.reshape((5_000, 50, 2))[:, :, 0] == 0, axis=0)

    assert missing_rate_by_time[-1] > missing_rate_by_time[0]
    assert missing_rate_by_time[-1] > missing_rate_by_time[10]


def test_lotka_volterra_time_mar_decreasing_masks_earlier_times_more() -> None:
    x_full = np.ones((5_000, 100))
    mask = LotkaVolterraTimeMARMask(
        missing_fraction=0.4,
        mode="decreasing",
    ).generate(x_full, theta=None, rng=np.random.default_rng(123))
    missing_rate_by_time = np.mean(mask.reshape((5_000, 50, 2))[:, :, 0] == 0, axis=0)

    assert missing_rate_by_time[0] > missing_rate_by_time[-1]
    assert missing_rate_by_time[0] > missing_rate_by_time[-11]


def test_lotka_volterra_log_total_mnar_masks_high_total_times_more() -> None:
    totals = np.linspace(1.0, 100.0, 50)
    states = np.stack([0.7 * totals, 0.3 * totals], axis=1)
    x_full = np.tile(states.reshape(1, -1), (5_000, 1))
    mask = LotkaVolterraLogTotalMNARMask(missing_fraction=0.8).generate(
        x_full,
        theta=None,
        rng=np.random.default_rng(123),
    )
    missing_rate_by_time = np.mean(mask.reshape((5_000, 50, 2))[:, :, 0] == 0, axis=0)

    assert missing_rate_by_time[-1] > missing_rate_by_time[0]
    assert missing_rate_by_time[-1] > missing_rate_by_time[10]


@pytest.mark.parametrize(
    "mask_generator",
    [
        LotkaVolterraTimeBlockMCARMask(missing_fraction=0.25, block_size=5),
        LotkaVolterraTimeMARMask(missing_fraction=0.25),
        LotkaVolterraLogTotalMNARMask(missing_fraction=0.25),
    ],
)
def test_lotka_volterra_masks_are_compatible_with_lotka_volterra_simulator(mask_generator) -> None:
    rng = np.random.default_rng(123)
    simulator = LotkaVolterraSimulator(num_timepoints=12, days=5.0)
    theta = simulator.sample_theta(3, rng)
    x_full = simulator.simulate(theta, rng)
    mask = mask_generator.generate(x_full, theta, rng)

    assert mask.shape == x_full.shape
    assert mask.dtype == np.int8
    _assert_binary(mask)
