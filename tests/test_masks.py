import numpy as np
import pytest

from gapsbi.masks import BlockMCARMask, PointMCARMask


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
