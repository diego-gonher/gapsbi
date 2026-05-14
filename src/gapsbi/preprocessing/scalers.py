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
