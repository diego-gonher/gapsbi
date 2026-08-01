from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ThetaScaler = MinMaxScaler | StandardScaler


def infer_theta_transform(problem_name: str | None) -> str:
    """Return the default theta transform for a given problem name."""
    if problem_name is not None and problem_name.lower() in {
        "glm",
        "glm_raw",
        "lotka_volterra",
    }:
        return "standard"
    return "minmax_minus_one_one"


def theta_scaling_metadata(transform: str) -> dict[str, Any]:
    """Return serializable metadata for a theta transform."""
    if transform == "standard":
        scaler = "standard"
    elif transform == "minmax_minus_one_one":
        scaler = "minmax"
    else:
        raise ValueError(
            "Invalid theta transform. Expected one of "
            "{'minmax_minus_one_one', 'standard'}, got "
            f"{transform!r}."
        )
    return {
        "transform": transform,
        "scaler": scaler,
        "fit_source": "theta_train",
    }


def scale_theta_train_val_test(
    theta_train: np.ndarray,
    theta_val: np.ndarray,
    theta_test: np.ndarray,
    transform: str = "minmax_minus_one_one",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ThetaScaler]:
    """Scale theta splits with a train-only fitted scaler."""
    theta_train_np = np.asarray(theta_train)
    theta_val_np = np.asarray(theta_val)
    theta_test_np = np.asarray(theta_test)

    if transform == "minmax_minus_one_one":
        theta_scaler: ThetaScaler = MinMaxScaler(feature_range=(-1, 1))
    elif transform == "standard":
        theta_scaler = StandardScaler()
    else:
        raise ValueError(
            "Invalid theta transform. Expected one of "
            "{'minmax_minus_one_one', 'standard'}, got "
            f"{transform!r}."
        )

    theta_train_scaled = theta_scaler.fit_transform(theta_train_np)
    theta_val_scaled = theta_scaler.transform(theta_val_np)
    theta_test_scaled = theta_scaler.transform(theta_test_np)

    return (
        torch.tensor(theta_train_scaled, dtype=torch.float32),
        torch.tensor(theta_val_scaled, dtype=torch.float32),
        torch.tensor(theta_test_scaled, dtype=torch.float32),
        theta_scaler,
    )


def make_scaled_theta_prior(
    problem_name: str,
    theta_dim: int,
    theta_train_scaled: torch.Tensor | None = None,
):
    """Build the NPE prior in the theta-scaled space."""
    if theta_dim <= 0:
        raise ValueError(f"theta_dim must be positive, got {theta_dim}.")
    if infer_theta_transform(problem_name) == "standard":
        loc = torch.zeros(theta_dim)
        covariance = torch.eye(theta_dim)
        if theta_train_scaled is not None and theta_train_scaled.shape[0] >= 2:
            theta_train_scaled = theta_train_scaled.detach().cpu().to(
                dtype=torch.float32
            )
            if theta_train_scaled.ndim != 2 or theta_train_scaled.shape[1] != theta_dim:
                raise ValueError(
                    f"theta_train_scaled must have shape (n, {theta_dim}), got "
                    f"{tuple(theta_train_scaled.shape)}."
                )
            centered = theta_train_scaled - theta_train_scaled.mean(dim=0, keepdim=True)
            covariance = centered.T @ centered / theta_train_scaled.shape[0]
            covariance = covariance + 1e-5 * torch.eye(theta_dim)
        return torch.distributions.MultivariateNormal(
            loc=loc,
            covariance_matrix=covariance,
        )

    from sbi.utils import BoxUniform

    return BoxUniform(low=-torch.ones(theta_dim), high=torch.ones(theta_dim))


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
