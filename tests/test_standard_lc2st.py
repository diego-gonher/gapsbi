from __future__ import annotations

import json
import numpy as np
import pytest

from gapsbi.evaluation.standard_lc2st import (
    assert_calibration_separate_from_evaluation,
    build_calibration_classes,
    corrected_permutation_p_value,
    run_standard_lc2st,
    split_paired_calibration_indices,
    write_standard_lc2st_result,
)


def test_corrected_permutation_p_value_includes_ties_and_extremes():
    assert corrected_permutation_p_value(2.0, np.array([1.0, 2.0, 3.0])) == 0.75
    assert corrected_permutation_p_value(4.0, np.array([1.0, 2.0, 3.0])) == 0.25


def test_paired_split_never_splits_class_pairs():
    train, holdout = split_paired_calibration_indices(13, 0.25, 9)
    assert not set(train) & set(holdout)
    assert set(train) | set(holdout) == set(range(13))


def test_calibration_evaluation_overlap_is_rejected():
    calibration = {"theta": np.array([[1.0]]), "x_full": np.array([[2.0, 3.0]]), "mask": np.array([[1, 0]])}
    observations = {"theta_true": calibration["theta"].copy(), "x_full": calibration["x_full"].copy(), "mask": calibration["mask"].copy()}
    with pytest.raises(ValueError, match="overlap"):
        assert_calibration_separate_from_evaluation(calibration, observations)


def test_class_construction_keeps_masks_and_rows_aligned():
    class Adapter:
        def sample(self, x_obs, mask, *, num_posterior_samples, seed):
            assert num_posterior_samples == 1 and seed == 7
            assert np.array_equal(mask, np.array([[1, 0], [0, 1]]))
            return np.array([[[10.0]], [[20.0]]])
    calibration = {"theta": np.array([[1.0], [2.0]]), "x_obs": np.array([[3.0, 0.0], [0.0, 4.0]]), "mask": np.array([[1, 0], [0, 1]])}
    q, prior, context = build_calibration_classes(Adapter(), calibration, 7)
    assert np.array_equal(q[:, 0], [10.0, 20.0])
    assert np.array_equal(prior[:, 0], [1.0, 2.0])
    assert np.array_equal(context, [[3.0, 0.0, 1.0, 0.0], [0.0, 4.0, 0.0, 1.0]])


def _result(seed: int = 13):
    rng = np.random.default_rng(4)
    n, d, c = 24, 2, 4
    prior = rng.normal(size=(n, d)).astype(np.float32)
    q = (prior + rng.normal(scale=0.1, size=(n, d))).astype(np.float32)
    context = rng.normal(size=(n, c)).astype(np.float32)
    eval_q = rng.normal(size=(2, 5, d)).astype(np.float32)
    eval_context = rng.normal(size=(2, c)).astype(np.float32)
    return run_standard_lc2st(calibration_q=q, calibration_prior=prior, calibration_context=context, evaluation_q=eval_q, evaluation_context=eval_context, reference_indices=np.array([8, 2]), classifier_seed=seed, num_permutations=3, classifier_max_iter=20, classifier_n_iter_no_change=5)


def test_runner_schema_reference_ids_pvalues_and_reproducibility(tmp_path):
    first, second = _result(), _result()
    assert first == second
    assert first["classifier"]["z_score"] is False
    assert [row["reference_index"] for row in first["reference_results"]] == [8, 2]
    for row in first["reference_results"]:
        assert len(row["null_local_mse"]) == 3
        assert row["p_value"] in {0.25, 0.5, 0.75, 1.0}
    output = tmp_path / "result.json"
    write_standard_lc2st_result(output, first)
    assert json.loads(output.read_text())["schema_name"] == "gapsbi_standard_lc2st"
    with pytest.raises(FileExistsError):
        write_standard_lc2st_result(output, first)
