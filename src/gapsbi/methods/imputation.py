from __future__ import annotations

import torch


def _validate_x_and_mask(
    x_obs_scaled: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if not torch.is_tensor(x_obs_scaled):
        raise ValueError("x_obs_scaled must be a torch.Tensor.")
    if not torch.is_tensor(mask):
        raise ValueError("mask must be a torch.Tensor.")
    if x_obs_scaled.ndim != 2:
        raise ValueError(
            f"x_obs_scaled must have shape (N, D), got ndim={x_obs_scaled.ndim}."
        )
    if mask.ndim != 2:
        raise ValueError(f"mask must have shape (N, D), got ndim={mask.ndim}.")
    if x_obs_scaled.shape != mask.shape:
        raise ValueError(
            f"x_obs_scaled and mask must have identical shapes, got "
            f"{tuple(x_obs_scaled.shape)} and {tuple(mask.shape)}."
        )


def zero_impute(
    x_obs_scaled: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Fill missing entries with 0.0 in scaled x-space."""
    _validate_x_and_mask(x_obs_scaled=x_obs_scaled, mask=mask)
    observed = mask.to(dtype=torch.bool)
    return torch.where(observed, x_obs_scaled, torch.zeros_like(x_obs_scaled))


def compute_observed_feature_means(
    x_obs_scaled_train: torch.Tensor,
    mask_train: torch.Tensor,
) -> torch.Tensor:
    """Compute feature means using only observed entries (mask == 1)."""
    _validate_x_and_mask(x_obs_scaled=x_obs_scaled_train, mask=mask_train)

    observed = mask_train.to(dtype=x_obs_scaled_train.dtype)
    observed_counts = observed.sum(dim=0)
    zero_observed = observed_counts == 0
    if zero_observed.any():
        missing_cols = torch.nonzero(zero_observed, as_tuple=False).flatten().tolist()
        raise ValueError(
            "Cannot compute observed feature means because some features have zero "
            f"observed entries: indices={missing_cols}."
        )

    observed_sums = (x_obs_scaled_train * observed).sum(dim=0)
    return observed_sums / observed_counts


def mean_impute(
    x_obs_scaled: torch.Tensor,
    mask: torch.Tensor,
    feature_means: torch.Tensor,
) -> torch.Tensor:
    """Fill missing entries in feature j with feature_means[j]."""
    _validate_x_and_mask(x_obs_scaled=x_obs_scaled, mask=mask)

    if not torch.is_tensor(feature_means):
        raise ValueError("feature_means must be a torch.Tensor.")
    if feature_means.ndim != 1:
        raise ValueError(
            f"feature_means must have shape (D,), got ndim={feature_means.ndim}."
        )
    if feature_means.shape[0] != x_obs_scaled.shape[1]:
        raise ValueError(
            f"feature_means must have length D={x_obs_scaled.shape[1]}, got "
            f"{feature_means.shape[0]}."
        )

    observed = mask.to(dtype=torch.bool)
    means = feature_means.to(dtype=x_obs_scaled.dtype, device=x_obs_scaled.device)
    means_row = means.unsqueeze(0).expand_as(x_obs_scaled)
    return torch.where(observed, x_obs_scaled, means_row)
