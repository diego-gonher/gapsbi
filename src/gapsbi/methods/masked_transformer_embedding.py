from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MaskedTransformerEmbeddingConfig:
    """Configuration for a lightweight mask-aware transformer NPE embedding."""

    x_dim: int = 1
    token_dim: int = 32
    context_dim: int = 64
    num_heads: int = 2
    num_layers: int = 1
    ff_multiplier: int = 2
    dropout: float = 0.0


class MaskedTransformerEmbedding(nn.Module):
    """Self-attention encoder for conditions `[x_zero_imputed, mask]`.

    Missing entries are zero-imputed before tokenization. The mask is included as a
    token channel, missing tokens are excluded as attention keys/values whenever a
    row has observed entries, and the final context uses learned attention pooling
    over observed tokens only.
    """

    def __init__(
        self,
        x_dim: int,
        token_dim: int = 32,
        context_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 1,
        ff_multiplier: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_positive("x_dim", x_dim)
        _validate_positive("token_dim", token_dim)
        _validate_positive("context_dim", context_dim)
        _validate_positive("num_heads", num_heads)
        _validate_positive("num_layers", num_layers)
        _validate_positive("ff_multiplier", ff_multiplier)
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads.")
        if dropout < 0.0:
            raise ValueError("dropout must be nonnegative.")

        self.x_dim = int(x_dim)
        self.token_dim = int(token_dim)
        self.context_dim = int(context_dim)

        self.value_embedding = nn.Linear(2, token_dim)
        self.feature_embedding = nn.Embedding(x_dim, token_dim)
        self.token_norm = nn.LayerNorm(token_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))

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
        self.pooling_score = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.Tanh(),
            nn.Linear(token_dim, 1),
        )
        self.output = nn.Sequential(
            nn.Linear(token_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        x_obs_scaled, mask = split_augmented_condition(condition, self.x_dim)
        tokens = self._tokenize(x_obs_scaled=x_obs_scaled, mask=mask)
        hidden = self.encoder(tokens, src_key_padding_mask=make_key_padding_mask(mask))
        pooled, _weights = self.attention_pool(hidden, mask=mask)
        return self.output(pooled)

    def _tokenize(self, x_obs_scaled: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        value_mask = torch.stack([x_obs_scaled, mask.to(dtype=x_obs_scaled.dtype)], dim=-1)
        tokens = self.value_embedding(value_mask)
        feature_ids = torch.arange(self.x_dim, device=x_obs_scaled.device)
        tokens = tokens + self.feature_embedding(feature_ids).unsqueeze(0)
        tokens = self.token_norm(tokens)
        return self.dropout(self.activation(tokens))

    def attention_pool(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool observed tokens with one learned attention query.

        Missing tokens receive exactly zero weight. Rows with no observed tokens
        return a zero pooled representation and zero weights.
        """
        _validate_tokens_mask(tokens=tokens, mask=mask)
        observed = mask.to(dtype=torch.bool, device=tokens.device)
        logits = self.pooling_score(tokens).squeeze(-1)
        masked_logits = logits.masked_fill(~observed, torch.finfo(logits.dtype).min)
        all_missing = ~observed.any(dim=1)
        if all_missing.any():
            masked_logits = masked_logits.clone()
            masked_logits[all_missing] = 0.0

        weights = torch.softmax(masked_logits, dim=1)
        weights = weights * observed.to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        return pooled, weights


def make_masked_transformer_embedding(
    config: MaskedTransformerEmbeddingConfig,
) -> MaskedTransformerEmbedding:
    return MaskedTransformerEmbedding(
        x_dim=config.x_dim,
        token_dim=config.token_dim,
        context_dim=config.context_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        ff_multiplier=config.ff_multiplier,
        dropout=config.dropout,
    )


def make_zero_imputed_mask_condition(
    x_obs_scaled: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return `[x_zero_imputed_scaled, mask]` for masked transformer NPE."""
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


def _validate_tokens_mask(tokens: torch.Tensor, mask: torch.Tensor) -> None:
    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape (N, D, H), got ndim={tokens.ndim}.")
    if mask.ndim != 2:
        raise ValueError(f"mask must have shape (N, D), got ndim={mask.ndim}.")
    if tokens.shape[:2] != mask.shape:
        raise ValueError(
            f"tokens and mask leading dimensions must match, got "
            f"{tuple(tokens.shape[:2])} and {tuple(mask.shape)}."
        )


def make_key_padding_mask(mask: torch.Tensor) -> torch.Tensor:
    """Return key padding mask with True for missing keys.

    PyTorch attention returns NaNs if all keys are masked. For all-missing rows,
    no keys are masked; learned attention pooling still returns a zero context.
    """
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
