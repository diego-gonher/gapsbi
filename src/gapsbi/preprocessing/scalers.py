from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def scale_theta_train_val_test(
    theta_train: np.ndarray,
    theta_val: np.ndarray,
    theta_test: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, MinMaxScaler]:
    """Scale theta splits with MinMaxScaler(-1, 1), fit on train only."""
    theta_train_np = np.asarray(theta_train)
    theta_val_np = np.asarray(theta_val)
    theta_test_np = np.asarray(theta_test)

    theta_scaler = MinMaxScaler(feature_range=(-1, 1))
    theta_train_scaled = theta_scaler.fit_transform(theta_train_np)
    theta_val_scaled = theta_scaler.transform(theta_val_np)
    theta_test_scaled = theta_scaler.transform(theta_test_np)

    return (
        torch.tensor(theta_train_scaled, dtype=torch.float32),
        torch.tensor(theta_val_scaled, dtype=torch.float32),
        torch.tensor(theta_test_scaled, dtype=torch.float32),
        theta_scaler,
    )


def scale_x_train_val_test(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    transform: str = "standard",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, StandardScaler, dict[str, Any]]:
    """Scale x splits fit on train only using standard or log1p+standard."""
    x_train_np = np.asarray(x_train)
    x_val_np = np.asarray(x_val)
    x_test_np = np.asarray(x_test)

    if transform == "standard":
        train_features = x_train_np
        val_features = x_val_np
        test_features = x_test_np
        pre_transform = "identity"
    elif transform == "log1p_standard":
        train_features = np.log1p(x_train_np)
        val_features = np.log1p(x_val_np)
        test_features = np.log1p(x_test_np)
        pre_transform = "log1p"
    else:
        raise ValueError(
            "Invalid transform. Expected one of {'standard', 'log1p_standard'}, "
            f"got {transform!r}."
        )

    x_scaler = StandardScaler()
    x_train_scaled = x_scaler.fit_transform(train_features)
    x_val_scaled = x_scaler.transform(val_features)
    x_test_scaled = x_scaler.transform(test_features)

    metadata = {
        "transform": transform,
        "pre_transform": pre_transform,
        "scaler": "standard",
    }

    return (
        torch.tensor(x_train_scaled, dtype=torch.float32),
        torch.tensor(x_val_scaled, dtype=torch.float32),
        torch.tensor(x_test_scaled, dtype=torch.float32),
        x_scaler,
        metadata,
    )


def infer_x_transform(problem_name: str) -> str:
    """Return the default x transform for a given problem name."""
    if problem_name.lower() == "ricker":
        return "log1p_standard"
    return "standard"


def fit_x_scaler_on_full_train(
    x_full_train: np.ndarray,
    transform: str = "standard",
) -> tuple[StandardScaler, dict[str, Any]]:
    """Fit x scaler on full train data only."""
    x_full_train_np = np.asarray(x_full_train)

    if transform == "standard":
        features_train = x_full_train_np
        pre_transform = "identity"
    elif transform == "log1p_standard":
        if np.any(x_full_train_np < 0):
            raise ValueError(
                "log1p_standard requires nonnegative x_full_train values for observed "
                "data."
            )
        features_train = np.log1p(x_full_train_np)
        pre_transform = "log1p"
    else:
        raise ValueError(
            "Invalid transform. Expected one of {'standard', 'log1p_standard'}, "
            f"got {transform!r}."
        )

    x_scaler = StandardScaler()
    x_scaler.fit(features_train)

    metadata = {
        "transform": transform,
        "pre_transform": pre_transform,
        "scaler": "standard",
        "fit_source": "x_full_train",
    }
    return x_scaler, metadata


def transform_x_obs_with_fitted_scaler(
    x_obs: np.ndarray,
    mask: np.ndarray,
    x_scaler: StandardScaler,
    transform: str = "standard",
) -> torch.Tensor:
    """Transform x_obs with a scaler fitted on x_full_train.

    Missing entries are replaced with finite placeholders before transformation
    and can be overwritten immediately after in scaled space by imputation.
    """
    x_obs_np = np.asarray(x_obs, dtype=np.float64)
    mask_np = np.asarray(mask)

    if x_obs_np.shape != mask_np.shape:
        raise ValueError(
            f"x_obs and mask must have identical shapes, got "
            f"{tuple(x_obs_np.shape)} and {tuple(mask_np.shape)}."
        )

    observed = mask_np.astype(bool)
    features = x_obs_np.copy()

    # Ensure missing locations are always finite placeholders before scaling.
    features[~observed] = 0.0

    if transform == "standard":
        pass
    elif transform == "log1p_standard":
        observed_values = features[observed]
        if np.any(observed_values < 0):
            raise ValueError(
                "log1p_standard requires nonnegative observed x_obs values."
            )
        features[observed] = np.log1p(observed_values)
    else:
        raise ValueError(
            "Invalid transform. Expected one of {'standard', 'log1p_standard'}, "
            f"got {transform!r}."
        )

    scaled = x_scaler.transform(features)
    return torch.tensor(scaled, dtype=torch.float32)


def scale_x_obs_train_val_test_from_full_train(
    x_full_train: np.ndarray,
    x_obs_train: np.ndarray,
    x_obs_val: np.ndarray,
    x_obs_test: np.ndarray,
    mask_train: np.ndarray,
    mask_val: np.ndarray,
    mask_test: np.ndarray,
    transform: str = "standard",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, StandardScaler, dict[str, Any]]:
    """Scale x_obs splits using scaler fit on x_full_train only."""
    x_scaler, metadata = fit_x_scaler_on_full_train(
        x_full_train=x_full_train,
        transform=transform,
    )
    x_train_scaled = transform_x_obs_with_fitted_scaler(
        x_obs=x_obs_train,
        mask=mask_train,
        x_scaler=x_scaler,
        transform=transform,
    )
    x_val_scaled = transform_x_obs_with_fitted_scaler(
        x_obs=x_obs_val,
        mask=mask_val,
        x_scaler=x_scaler,
        transform=transform,
    )
    x_test_scaled = transform_x_obs_with_fitted_scaler(
        x_obs=x_obs_test,
        mask=mask_test,
        x_scaler=x_scaler,
        transform=transform,
    )

    return x_train_scaled, x_val_scaled, x_test_scaled, x_scaler, metadata
