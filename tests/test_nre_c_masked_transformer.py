from __future__ import annotations

import torch

from gapsbi.methods.masked_transformer_embedding import (
    MaskedTransformerEmbedding,
    MaskedTransformerEmbeddingConfig,
    make_zero_imputed_mask_condition,
)
from gapsbi.methods.sbi_nre import make_masked_transformer_nre_classifier


def test_nre_c_uses_the_shared_canonical_npe_transformer():
    config = MaskedTransformerEmbeddingConfig(x_dim=3)
    builder = make_masked_transformer_nre_classifier(config)
    estimator = builder(torch.randn(4, 2), torch.randn(4, 6))
    encoder = estimator.embedding_net_x
    assert isinstance(encoder, MaskedTransformerEmbedding)
    assert (encoder.token_dim, encoder.context_dim) == (32, 64)
    assert len(encoder.encoder.layers) == 1
    assert encoder.encoder.layers[0].self_attn.num_heads == 2


def test_nre_c_transformer_encoder_handles_shared_mask_representation_and_gradients():
    config = MaskedTransformerEmbeddingConfig(x_dim=3)
    encoder = MaskedTransformerEmbedding(**config.__dict__)
    condition = make_zero_imputed_mask_condition(torch.randn(2, 3), torch.zeros(2, 3))
    output = encoder(condition)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in encoder.parameters())
