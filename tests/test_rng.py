import numpy as np
import pytest

from gapsbi.rng import make_rng, make_rngs, split_rng


def test_make_rng_reproducible() -> None:
    rng1 = make_rng(123)
    rng2 = make_rng(123)

    assert np.allclose(rng1.normal(size=10), rng2.normal(size=10))


def test_split_rng_reproducible() -> None:
    rng1 = make_rng(123)
    rng2 = make_rng(123)

    children1 = split_rng(rng1, 3)
    children2 = split_rng(rng2, 3)

    for child1, child2 in zip(children1, children2, strict=True):
        assert np.allclose(child1.normal(size=10), child2.normal(size=10))


def test_split_rng_children_are_distinct() -> None:
    rng = make_rng(123)
    children = split_rng(rng, 3)
    samples = [child.normal(size=10) for child in children]

    assert not np.allclose(samples[0], samples[1])
    assert not np.allclose(samples[1], samples[2])


def test_split_rng_negative_n_raises() -> None:
    with pytest.raises(ValueError, match="n must be nonnegative"):
        split_rng(make_rng(123), -1)


def test_make_rngs_reproducible_and_named() -> None:
    rngs1 = make_rngs(123, ["simulator", "mask", "split"])
    rngs2 = make_rngs(123, ["simulator", "mask", "split"])

    assert list(rngs1) == ["simulator", "mask", "split"]
    assert list(rngs2) == ["simulator", "mask", "split"]
    for name in rngs1:
        assert np.allclose(rngs1[name].normal(size=10), rngs2[name].normal(size=10))
