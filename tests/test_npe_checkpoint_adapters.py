"""Smoke and regression tests for read-only NPE checkpoint adapters."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from gapsbi.evaluation.checkpoint_adapters import load_npe_checkpoint_adapter
from gapsbi.methods.imputation import zero_impute
from gapsbi.methods.mask_augmentation import make_zero_imputed_mask_augmented_input
from gapsbi.methods.masked_transformer_embedding import make_zero_imputed_mask_condition
from gapsbi.preprocessing.scalers import transform_x_obs_with_fitted_scaler
from gapsbi.utils.seeding import set_all_seeds


ROOT = Path(__file__).parents[1]
TASK_PATHS = {
    "oup": "oup/mcar/oup_mcar_eps010_seed123.h5",
    "glm": "glm/mcar/glm_raw_mcar_eps010_seed123.h5",
    "glu": "glu/mcar/glu_mcar_eps010_seed123.h5",
    "lotka_volterra": "lotka_volterra/mcar/lotka_volterra_time_block_mcar_eps010_seed123.h5",
}


def checkpoint_path(method: str, task: str, budget: str = "low", seed: int = 101) -> Path:
    stem = TASK_PATHS[task].rsplit("/", 1)[1].removesuffix("_seed123.h5")
    return ROOT / f"outputs_{budget}_sim_budget" / method / task / "mcar" / stem / f"seed_{seed}/model_checkpoint.pt"


def dataset_path(task: str) -> Path:
    return ROOT / "data/canonical_v1" / TASK_PATHS[task]


def require_artifacts(method: str, task: str) -> None:
    if not checkpoint_path(method, task).is_file() or not dataset_path(task).is_file():
        pytest.skip("ignored checkpoint/dataset artifacts are unavailable")


@pytest.mark.parametrize(
    "method",
    [
        "npe_zero_imputation",
        "npe_mask_augmentation",
        "npe_learned_constant_imputation",
        "npe_probabilistic_learned_imputation",
        "npe_masked_transformer_embedding",
    ],
)
@pytest.mark.parametrize("task", tuple(TASK_PATHS))
def test_adapter_reconstructs_and_samples_reproducibly(method: str, task: str) -> None:
    require_artifacts(method, task)
    with h5py.File(dataset_path(task), "r") as h5:
        x_obs, mask = h5["test/x_obs"][:2], h5["test/mask"][:2]
    adapter = load_npe_checkpoint_adapter(checkpoint_path(method, task), repository_root=ROOT)
    prepared = adapter.prepare_input(x_obs, mask)
    expected_dim = x_obs.shape[1] * (
        2 if method in {"npe_mask_augmentation", "npe_masked_transformer_embedding"} else 1
    )
    assert prepared.shape == (2, expected_dim)
    x_scaled = transform_x_obs_with_fitted_scaler(
        x_obs=x_obs,
        mask=mask,
        x_scaler=adapter.checkpoint["x_scaler"],
        transform=adapter.checkpoint["x_scaling_metadata"]["transform"],
    )
    mask_tensor = torch.as_tensor(mask, dtype=torch.float32)
    if method == "npe_zero_imputation":
        expected = zero_impute(x_scaled, mask_tensor)
    elif method == "npe_mask_augmentation":
        expected = make_zero_imputed_mask_augmented_input(x_scaled, mask_tensor)
    elif method == "npe_learned_constant_imputation":
        assert adapter.imputer is not None
        expected = adapter.imputer(x_scaled, mask_tensor)
    elif method == "npe_probabilistic_learned_imputation":
        assert adapter.imputer is not None
        set_all_seeds(44)
        expected = adapter.imputer(x_scaled, mask_tensor).completed_x
        set_all_seeds(44)
        prepared = adapter.prepare_input(x_obs, mask)
        np.testing.assert_array_equal(prepared.cpu().numpy(), expected.detach().numpy())
        prepared = None
    else:
        expected = make_zero_imputed_mask_condition(x_scaled, mask_tensor)
    if prepared is not None:
        np.testing.assert_array_equal(prepared.cpu().numpy(), expected.detach().numpy())
    first = adapter.sample(x_obs, mask, num_posterior_samples=2, seed=71_001)
    second = adapter.sample(x_obs, mask, num_posterior_samples=2, seed=71_001)
    assert first.shape == (2, 2, adapter.checkpoint["theta_dim"])
    assert np.all(np.isfinite(first))
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    ("method", "task", "budget", "seed"),
    [
        ("npe_zero_imputation", "oup", "mid", 202),
        ("npe_zero_imputation", "glm", "high", 303),
        ("npe_zero_imputation", "glu", "mid", 404),
        ("npe_zero_imputation", "lotka_volterra", "high", 505),
        ("npe_mask_augmentation", "oup", "high", 202),
        ("npe_mask_augmentation", "glm", "mid", 303),
        ("npe_mask_augmentation", "glu", "high", 404),
        ("npe_mask_augmentation", "lotka_volterra", "mid", 505),
        ("npe_learned_constant_imputation", "oup", "mid", 202),
        ("npe_learned_constant_imputation", "glm", "high", 303),
        ("npe_probabilistic_learned_imputation", "glu", "mid", 404),
        ("npe_probabilistic_learned_imputation", "lotka_volterra", "high", 505),
        ("npe_masked_transformer_embedding", "oup", "high", 202),
        ("npe_masked_transformer_embedding", "glm", "mid", 303),
    ],
)
def test_adapter_loads_other_saved_budgets_and_estimator_seeds(
    method: str, task: str, budget: str, seed: int
) -> None:
    checkpoint = checkpoint_path(method, task, budget, seed)
    if not checkpoint.is_file() or not dataset_path(task).is_file():
        pytest.skip("ignored checkpoint/dataset artifacts are unavailable")
    adapter = load_npe_checkpoint_adapter(checkpoint, repository_root=ROOT)
    assert adapter.method == method
    assert adapter.problem == task
    assert adapter.checkpoint["seed"] == seed


def test_adapter_matches_canonical_glu_preprocessing() -> None:
    method, task = "npe_zero_imputation", "glu"
    require_artifacts(method, task)
    run = checkpoint_path(method, task).parent
    with h5py.File(dataset_path(task), "r") as dataset:
        x_obs, mask = dataset["test/x_obs"][:1], dataset["test/mask"][:1]
    adapter = load_npe_checkpoint_adapter(run / "model_checkpoint.pt", repository_root=ROOT)
    expected = zero_impute(
        transform_x_obs_with_fitted_scaler(
            x_obs=x_obs,
            mask=mask,
            x_scaler=adapter.checkpoint["x_scaler"],
            transform=adapter.checkpoint["x_scaling_metadata"]["transform"],
        ),
        torch.as_tensor(mask, dtype=torch.float32),
    )
    np.testing.assert_array_equal(adapter.prepare_input(x_obs, mask).cpu().numpy(), expected.numpy())


def test_adapter_rejects_nonzero_missing_values() -> None:
    method, task = "npe_mask_augmentation", "glu"
    require_artifacts(method, task)
    with h5py.File(dataset_path(task), "r") as h5:
        x_obs, mask = h5["test/x_obs"][:1], h5["test/mask"][:1]
    missing = np.argwhere(mask == 0)
    if missing.size == 0:
        pytest.skip("selected deterministic fixture unexpectedly has no missing value")
    x_obs[tuple(missing[0])] = 1.0
    adapter = load_npe_checkpoint_adapter(checkpoint_path(method, task), repository_root=ROOT)
    with pytest.raises(ValueError, match="zero at missing"):
        adapter.prepare_input(x_obs, mask)
