from __future__ import annotations

import torch

from gapsbi.methods.imputation import zero_impute


def make_zero_imputed_mask_augmented_input(
    x_obs_scaled: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Create Wang-style augmented input [zero_imputed_x, mask]."""
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

    x_imputed = zero_impute(x_obs_scaled=x_obs_scaled, mask=mask)
    mask_float = mask.to(dtype=torch.float32, device=x_imputed.device)
    return torch.cat([x_imputed, mask_float], dim=-1)
