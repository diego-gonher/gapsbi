import pytest

torch = pytest.importorskip("torch")

from gapsbi.methods.masked_transformer_embedding import (
    MaskedTransformerEmbedding,
    MaskedTransformerEmbeddingConfig,
    make_key_padding_mask,
    make_masked_transformer_embedding,
    make_zero_imputed_mask_condition,
    split_augmented_condition,
)


def test_make_zero_imputed_mask_condition_shape_and_content() -> None:
    x_obs_scaled = torch.tensor([[1.0, -2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    condition = make_zero_imputed_mask_condition(x_obs_scaled=x_obs_scaled, mask=mask)

    assert condition.shape == (2, 6)
    assert torch.equal(
        condition[:, :3],
        torch.tensor([[1.0, 0.0, 3.0], [0.0, 5.0, 0.0]]),
    )
    assert torch.equal(condition[:, 3:], mask)


def test_split_augmented_condition_roundtrip() -> None:
    x_obs_scaled = torch.randn(4, 5)
    mask = (torch.rand(4, 5) > 0.5).float()
    condition = make_zero_imputed_mask_condition(x_obs_scaled=x_obs_scaled, mask=mask)

    x_split, mask_split = split_augmented_condition(condition, x_dim=5)

    assert x_split.shape == (4, 5)
    assert mask_split.shape == (4, 5)
    assert torch.equal(mask_split, mask)


def test_attention_pooling_weights_sum_to_one_over_observed_tokens() -> None:
    torch.manual_seed(1)
    embedding = MaskedTransformerEmbedding(
        x_dim=3,
        token_dim=4,
        context_dim=5,
        num_heads=2,
        num_layers=1,
    )
    tokens = torch.randn(2, 3, 4)
    mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

    _pooled, weights = embedding.attention_pool(tokens, mask=mask)

    assert torch.allclose((weights * mask).sum(dim=1), torch.ones(2))
    assert torch.equal(weights[mask == 0.0], torch.zeros_like(weights[mask == 0.0]))


def test_attention_pooling_missing_tokens_do_not_affect_pooled_representation() -> None:
    torch.manual_seed(2)
    embedding = MaskedTransformerEmbedding(
        x_dim=3,
        token_dim=4,
        context_dim=5,
        num_heads=2,
        num_layers=1,
    )
    tokens = torch.randn(1, 3, 4)
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    changed = tokens.clone()
    changed[:, 1, :] = torch.tensor([100.0, -100.0, 50.0, -50.0])

    pooled, weights = embedding.attention_pool(tokens, mask=mask)
    pooled_changed, weights_changed = embedding.attention_pool(changed, mask=mask)

    assert torch.equal(weights[:, 1], torch.zeros_like(weights[:, 1]))
    assert torch.equal(weights_changed[:, 1], torch.zeros_like(weights_changed[:, 1]))
    assert torch.allclose(pooled, pooled_changed)


def test_attention_pooling_all_missing_rows_return_zero_and_finite() -> None:
    torch.manual_seed(3)
    embedding = MaskedTransformerEmbedding(
        x_dim=3,
        token_dim=4,
        context_dim=5,
        num_heads=2,
        num_layers=1,
    )
    tokens = torch.randn(2, 3, 4)
    mask = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]])

    pooled, weights = embedding.attention_pool(tokens, mask=mask)

    assert torch.equal(pooled[0], torch.zeros(4))
    assert torch.equal(weights[0], torch.zeros(3))
    assert torch.isfinite(pooled).all()
    assert torch.isfinite(weights).all()


def test_key_padding_mask_does_not_mask_all_tokens_for_all_missing_rows() -> None:
    mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])

    key_padding_mask = make_key_padding_mask(mask)

    assert torch.equal(key_padding_mask[0], torch.tensor([False, True, False]))
    assert torch.equal(key_padding_mask[1], torch.tensor([False, False, False]))


def test_masked_transformer_embedding_forward_shape_and_finite() -> None:
    torch.manual_seed(1)
    condition = torch.tensor(
        [
            [1.0, 0.0, 2.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    embedding = MaskedTransformerEmbedding(
        x_dim=3,
        token_dim=8,
        context_dim=5,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    )

    out = embedding(condition)

    assert out.shape == (2, 5)
    assert torch.isfinite(out).all()


def test_attention_pooling_parameters_receive_gradients() -> None:
    torch.manual_seed(4)
    condition = torch.tensor(
        [
            [1.0, 0.0, 2.0, 1.0, 0.0, 1.0],
            [0.5, -1.0, 0.0, 1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    embedding = MaskedTransformerEmbedding(
        x_dim=3,
        token_dim=8,
        context_dim=5,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    )

    loss = embedding(condition).sum()
    loss.backward()

    pooling_params = list(embedding.pooling_score.parameters())
    assert pooling_params
    assert all(param.grad is not None for param in pooling_params)
    assert any(torch.any(param.grad != 0.0) for param in pooling_params)


def test_make_masked_transformer_embedding_factory() -> None:
    embedding = make_masked_transformer_embedding(
        MaskedTransformerEmbeddingConfig(
            x_dim=4,
            token_dim=8,
            context_dim=6,
            num_heads=2,
            num_layers=1,
        )
    )

    assert isinstance(embedding, MaskedTransformerEmbedding)
