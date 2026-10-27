from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from gapsbi.methods.mask_augmentation import make_zero_imputed_mask_augmented_input
from gapsbi.methods.sbi_nre import (
    prepare_mask_augmented_nre_input,
    prepare_zero_imputed_nre_input,
)


def _load_evaluator():
    path = Path("experiments/nre_c_zero_imputation/evaluate.py")
    spec = importlib.util.spec_from_file_location("nre_c_zero_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_zero_imputed_nre_input_uses_scaled_space_and_never_appends_mask():
    scaler = StandardScaler().fit(np.array([[0.0, 10.0], [2.0, 14.0]]))
    x_obs = np.array([[2.0, 999.0]])
    mask = np.array([[1, 0]])
    result = prepare_zero_imputed_nre_input(
        x_obs=x_obs, mask=mask, x_scaler=scaler, transform="standard"
    ).numpy()
    assert result.shape == (1, 2)
    assert np.isclose(result[0, 0], 1.0)
    assert result[0, 1] == 0.0


def test_mask_augmented_nre_input_matches_canonical_npe_representation():
    scaler = StandardScaler().fit(np.array([[0.0, 10.0], [2.0, 14.0]]))
    x_obs = np.array([[2.0, 999.0]])
    mask = np.array([[1, 0]], dtype=np.float32)
    result = prepare_mask_augmented_nre_input(
        x_obs=x_obs, mask=mask, x_scaler=scaler, transform="standard"
    )
    scaled = np.array([[1.0, -6.0]], dtype=np.float32)
    expected = make_zero_imputed_mask_augmented_input(
        x_obs_scaled=torch.tensor(scaled), mask=torch.tensor(mask),
    )
    assert result.shape == (1, 4)
    assert np.array_equal(result.numpy(), expected.numpy())


def test_select_reference_positions_supports_sparse_canonical_ids():
    evaluator = _load_evaluator()
    positions, ids = evaluator.select_reference_positions(
        {"x_obs": np.zeros((3, 2)), "reference_indices": np.array([0, 3, 8])}, [8, 0]
    )
    assert positions.tolist() == [2, 0]
    assert ids.tolist() == [8, 0]


def test_select_reference_positions_rejects_unavailable_id():
    evaluator = _load_evaluator()
    try:
        evaluator.select_reference_positions({"x_obs": np.zeros((1, 2))}, [3])
    except ValueError as error:
        assert "not uniquely available" in str(error)
    else:
        raise AssertionError("Unavailable reference ID was accepted.")
