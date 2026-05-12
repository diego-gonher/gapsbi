import numpy as np
import pytest

from gapsbi.masks import BlockMCARMask, PointMCARMask, SelfCensoringMNARMask
from gapsbi.simulators import GLMSimulator, GLUSimulator, OUPSimulator, RickerSimulator


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
