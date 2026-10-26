import numpy as np
import pytest

from gapsbi.evaluation.lc2st import (
    derangement,
    farthest_reference_indices,
    mask_augmented_context,
    select_one_finite_draw_per_observation,
)


def test_mask_augmented_context_preserves_order_and_validates_mask():
    x = np.array([[1.0, 0.0], [2.0, 3.0]])
    mask = np.array([[1, 0], [1, 1]])
    result = mask_augmented_context(x, mask)
    assert np.array_equal(result, np.array([[1.0, 0.0, 1.0, 0.0], [2.0, 3.0, 1.0, 1.0]]))
    with pytest.raises(ValueError, match="binary"):
        mask_augmented_context(x, np.array([[1, 2], [1, 1]]))


def test_select_one_draw_is_reproducible_and_skips_nonfinite_draws():
    samples = np.arange(24, dtype=float).reshape(2, 4, 3)
    samples[0, 0] = np.nan
    selected_a, draws_a = select_one_finite_draw_per_observation(samples, 7)
    selected_b, draws_b = select_one_finite_draw_per_observation(samples, 7)
    assert np.array_equal(draws_a, draws_b)
    assert np.array_equal(selected_a, selected_b)
    assert draws_a[0] != 0
    samples[1] = np.nan
    with pytest.raises(ValueError, match="No finite posterior"):
        select_one_finite_draw_per_observation(samples, 7)


def test_derangement_is_reproducible_and_has_no_fixed_points():
    first = derangement(10, 17)
    assert np.array_equal(first, derangement(10, 17))
    assert np.array_equal(np.sort(first), np.arange(10))
    assert np.all(first != np.arange(10))


def test_farthest_reference_indices_never_selects_own_block():
    samples = np.array([[[0.0], [0.0]], [[2.0], [2.0]], [[10.0], [10.0]]])
    indices = farthest_reference_indices(samples)
    assert np.array_equal(indices, np.array([2, 2, 0]))
    assert np.all(indices != np.arange(3))
