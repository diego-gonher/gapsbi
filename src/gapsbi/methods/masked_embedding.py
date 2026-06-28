from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


MaskedEmbeddingType = Literal["masked_pooling", "masked_attention"]


@dataclass(frozen=True)
class MaskedEmbeddingConfig:
    """Configuration for lightweight mask-aware NPE embedding networks."""

    embedding_type: MaskedEmbeddingType = "masked_attention"
    x_dim: int = 1
    token_dim: int = 32
    context_dim: int = 64
    num_heads: int = 2
    num_layers: int = 1
    ff_multiplier: int = 2
    dropout: float = 0.0
    mask_summary_dim: int = 16


class MaskedPoolingEmbedding(nn.Module):
    """Small mask-aware pooling encoder for conditions `[x_zero_imputed, mask]`.

    Each feature is represented as a scalar token with a learned feature embedding.
    Observed tokens are pooled with a masked mean, while a separate mask-summary MLP
    preserves information in the missingness pattern.
    """

    def __init__(
        self,
        x_dim: int,
        token_dim: int = 32,
        context_dim: int = 64,
        mask_summary_dim: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_positive("x_dim", x_dim)
        _validate_positive("token_dim", token_dim)
        _validate_positive("context_dim", context_dim)
        _validate_positive("mask_summary_dim", mask_summary_dim)
        if dropout < 0.0:
            raise ValueError("dropout must be nonnegative.")

        self.x_dim = int(x_dim)
        self.token_dim = int(token_dim)
        self.context_dim = int(context_dim)
        self.mask_summary_dim = int(mask_summary_dim)

        self.value_embedding = nn.Linear(2, token_dim)
        self.feature_embedding = nn.Embedding(x_dim, token_dim)
        self.token_norm = nn.LayerNorm(token_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.mask_summary = nn.Sequential(
            nn.Linear(x_dim, mask_summary_dim),
            nn.GELU(),
            nn.Linear(mask_summary_dim, mask_summary_dim),
        )
        self.output = nn.Sequential(
            nn.Linear(token_dim + mask_summary_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        x_obs_scaled, mask = split_augmented_condition(condition, self.x_dim)
        tokens = self._tokenize(x_obs_scaled=x_obs_scaled, mask=mask)
        pooled = masked_mean(tokens, mask=mask)
        mask_summary = self.mask_summary(mask.to(dtype=tokens.dtype))
        return self.output(torch.cat([pooled, mask_summary], dim=-1))

    def _tokenize(self, x_obs_scaled: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        value_mask = torch.stack([x_obs_scaled, mask.to(dtype=x_obs_scaled.dtype)], dim=-1)
        tokens = self.value_embedding(value_mask)
        feature_ids = torch.arange(self.x_dim, device=x_obs_scaled.device)
        tokens = tokens + self.feature_embedding(feature_ids).unsqueeze(0)
        tokens = self.token_norm(tokens)
        return self.dropout(self.activation(tokens))


class MaskedAttentionEmbedding(MaskedPoolingEmbedding):
    """Small self-attention encoder for conditions `[x_zero_imputed, mask]`.

    Missing entries are excluded as attention keys/values for rows with at least
    one observed feature. The final representation still includes a separate
    mask-summary path, so MAR/MNAR missingness patterns are not discarded.
    """

    def __init__(
        self,
        x_dim: int,
        token_dim: int = 32,
        context_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 1,
        ff_multiplier: int = 2,
        mask_summary_dim: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            x_dim=x_dim,
            token_dim=token_dim,
            context_dim=context_dim,
            mask_summary_dim=mask_summary_dim,
            dropout=dropout,
        )
        _validate_positive("num_heads", num_heads)
        _validate_positive("num_layers", num_layers)
        _validate_positive("ff_multiplier", ff_multiplier)
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads.")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=ff_multiplier * token_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        x_obs_scaled, mask = split_augmented_condition(condition, self.x_dim)
        tokens = self._tokenize(x_obs_scaled=x_obs_scaled, mask=mask)
        key_padding_mask = make_key_padding_mask(mask)
        hidden = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        pooled = masked_mean(hidden, mask=mask)
        mask_summary = self.mask_summary(mask.to(dtype=hidden.dtype))
        return self.output(torch.cat([pooled, mask_summary], dim=-1))


def make_masked_embedding(config: MaskedEmbeddingConfig) -> nn.Module:
    if config.embedding_type == "masked_pooling":
        return MaskedPoolingEmbedding(
            x_dim=config.x_dim,
            token_dim=config.token_dim,
            context_dim=config.context_dim,
            mask_summary_dim=config.mask_summary_dim,
            dropout=config.dropout,
        )
    if config.embedding_type == "masked_attention":
        return MaskedAttentionEmbedding(
            x_dim=config.x_dim,
            token_dim=config.token_dim,
            context_dim=config.context_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            ff_multiplier=config.ff_multiplier,
            mask_summary_dim=config.mask_summary_dim,
            dropout=config.dropout,
        )
    raise ValueError(
        "embedding_type must be one of {'masked_pooling', 'masked_attention'}, "
        f"got {config.embedding_type!r}."
    )


def make_zero_imputed_mask_condition(
    x_obs_scaled: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return `[x_zero_imputed_scaled, mask]` for masked embedding NPE."""
    _validate_x_mask(x_obs_scaled=x_obs_scaled, mask=mask)
    mask_float = mask.to(dtype=x_obs_scaled.dtype, device=x_obs_scaled.device)
    x_zero = torch.where(
        mask_float.to(dtype=torch.bool),
        x_obs_scaled,
        torch.zeros_like(x_obs_scaled),
    )
    return torch.cat([x_zero, mask_float], dim=-1)


def split_augmented_condition(
    condition: torch.Tensor,
    x_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(condition):
        raise ValueError("condition must be a torch.Tensor.")
    if condition.ndim != 2:
        raise ValueError(f"condition must have shape (N, 2D), got ndim={condition.ndim}.")
    if condition.shape[1] != 2 * int(x_dim):
        raise ValueError(
            f"condition must have feature dimension 2 * x_dim={2 * int(x_dim)}, "
            f"got {condition.shape[1]}."
        )
    x_obs_scaled = condition[:, :x_dim]
    mask = condition[:, x_dim:]
    return x_obs_scaled, mask


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape (N, D, H), got ndim={tokens.ndim}.")
    if mask.ndim != 2:
        raise ValueError(f"mask must have shape (N, D), got ndim={mask.ndim}.")
    if tokens.shape[:2] != mask.shape:
        raise ValueError(
            f"tokens and mask leading dimensions must match, got "
            f"{tuple(tokens.shape[:2])} and {tuple(mask.shape)}."
        )
    weights = mask.to(dtype=tokens.dtype, device=tokens.device).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (tokens * weights).sum(dim=1) / denom


def make_key_padding_mask(mask: torch.Tensor) -> torch.Tensor:
    """Return a Transformer key padding mask that is robust to all-missing rows."""
    if mask.ndim != 2:
        raise ValueError(f"mask must have shape (N, D), got ndim={mask.ndim}.")
    observed = mask.to(dtype=torch.bool)
    key_padding_mask = ~observed
    all_missing = ~observed.any(dim=1)
    if all_missing.any():
        key_padding_mask = key_padding_mask.clone()
        key_padding_mask[all_missing] = False
    return key_padding_mask


def _validate_x_mask(x_obs_scaled: torch.Tensor, mask: torch.Tensor) -> None:
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


def _validate_positive(name: str, value: int) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
