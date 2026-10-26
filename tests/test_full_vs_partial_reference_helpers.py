import numpy as np
import pytest

from gapsbi.analysis.full_vs_partial_references import (
    bootstrap_interval,
    cross_sample_duplicate_summary,
    matched_reference_indices,
)


def test_sparse_matching_uses_original_ids():
    full = {"theta_true": np.arange(20).reshape(10, 2), "x_full": np.arange(30).reshape(10, 3)}
    partial = {"theta_true": full["theta_true"][[0, 3]], "x_full": full["x_full"][[0, 3]], "reference_indices": np.array([0, 3])}
    assert np.array_equal(matched_reference_indices(full, partial), [0, 3])
    partial["x_full"] = partial["x_full"].copy(); partial["x_full"][0, 0] = -1
    with pytest.raises(ValueError, match="x_full"):
        matched_reference_indices(full, partial)


def test_bootstrap_interval_is_deterministic_and_respects_statistic():
    values = np.array([1., 2., 9.])
    first = bootstrap_interval(values, statistic="median", seed=11, draws=100)
    assert first == bootstrap_interval(values, statistic="median", seed=11, draws=100)
    assert first[0] == 2.


def test_cross_sample_duplicate_summary_detects_copied_rows_only():
    first = np.array([[1., 2.], [3., 4.], [3., 4.]])
    copied = first.copy()
    independent = np.array([[1.1, 2.], [3.1, 4.], [5., 6.]])
    copied_summary = cross_sample_duplicate_summary(first, copied)
    assert copied_summary["shared_unique_rows"] == 2
    assert copied_summary["shared_first_row_count"] == 3
    assert cross_sample_duplicate_summary(first, independent)["shared_unique_rows"] == 0
