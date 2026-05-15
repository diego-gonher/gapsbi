import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gapsbi.methods.imputation import compute_observed_feature_means, mean_impute, zero_impute
from gapsbi.preprocessing.scalers import (
    fit_x_scaler_on_full_train,
    scale_x_obs_train_val_test_from_full_train,
)


def test_x_obs_scaling_uses_scaler_fit_on_x_full_train_standard() -> None:
    x_full_train = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
    x_obs_train = np.array([[10.0, 0.0], [0.0, 40.0]], dtype=np.float64)
    x_obs_val = np.array([[30.0, 0.0]], dtype=np.float64)
    x_obs_test = np.array([[0.0, 20.0]], dtype=np.float64)

    mask_train = np.array([[1, 0], [0, 1]], dtype=np.float64)
    mask_val = np.array([[1, 0]], dtype=np.float64)
    mask_test = np.array([[0, 1]], dtype=np.float64)

    x_train_s, x_val_s, x_test_s, x_scaler, metadata = (
        scale_x_obs_train_val_test_from_full_train(
            x_full_train=x_full_train,
            x_obs_train=x_obs_train,
            x_obs_val=x_obs_val,
            x_obs_test=x_obs_test,
            mask_train=mask_train,
            mask_val=mask_val,
            mask_test=mask_test,
            transform="standard",
        )
    )

    assert np.allclose(x_scaler.mean_, np.array([20.0, 30.0]))
    assert metadata["fit_source"] == "x_full_train"
    assert x_train_s.shape == (2, 2)
    assert x_val_s.shape == (1, 2)
    assert x_test_s.shape == (1, 2)


def test_ricker_log1p_path_handles_missing_placeholders_and_imputation() -> None:
    x_full_train = np.array([[0.0, 3.0], [7.0, 15.0]], dtype=np.float64)
    x_obs_train = np.array([[0.0, np.nan], [np.nan, 15.0]], dtype=np.float64)
    x_obs_val = np.array([[7.0, np.nan]], dtype=np.float64)
    x_obs_test = np.array([[np.nan, 3.0]], dtype=np.float64)

    mask_train = np.array([[1, 0], [0, 1]], dtype=np.float64)
    mask_val = np.array([[1, 0]], dtype=np.float64)
    mask_test = np.array([[0, 1]], dtype=np.float64)

    x_train_s, x_val_s, x_test_s, x_scaler, _metadata = (
        scale_x_obs_train_val_test_from_full_train(
            x_full_train=x_full_train,
            x_obs_train=x_obs_train,
            x_obs_val=x_obs_val,
            x_obs_test=x_obs_test,
            mask_train=mask_train,
            mask_val=mask_val,
            mask_test=mask_test,
            transform="log1p_standard",
        )
    )

    # Scaler must be fit on log1p(full-train), not observed-only data.
    expected_scaler, _ = fit_x_scaler_on_full_train(
        x_full_train=x_full_train,
        transform="log1p_standard",
    )
    assert np.allclose(x_scaler.mean_, expected_scaler.mean_)

    # Missing placeholders should not leak NaNs into scaled outputs.
    assert torch.isfinite(x_train_s).all()
    assert torch.isfinite(x_val_s).all()
    assert torch.isfinite(x_test_s).all()

    # Imputation happens in scaled space.
    mask_train_t = torch.tensor(mask_train, dtype=torch.float32)
    feature_means = compute_observed_feature_means(
        x_obs_scaled_train=x_train_s,
        mask_train=mask_train_t,
    )
    x_train_zero = zero_impute(x_obs_scaled=x_train_s, mask=mask_train_t)
    x_train_mean = mean_impute(
        x_obs_scaled=x_train_s,
        mask=mask_train_t,
        feature_means=feature_means,
    )
    assert x_train_zero.shape == x_train_s.shape
    assert x_train_mean.shape == x_train_s.shape
