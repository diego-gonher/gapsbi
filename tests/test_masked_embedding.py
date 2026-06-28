import pytest

torch = pytest.importorskip("torch")

from gapsbi.methods.masked_embedding import (
    MaskedAttentionEmbedding,
    MaskedEmbeddingConfig,
    MaskedPoolingEmbedding,
    make_key_padding_mask,
    make_masked_embedding,
    make_zero_imputed_mask_condition,
    masked_mean,
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


def test_masked_mean_ignores_missing_and_handles_all_missing() -> None:
    tokens = torch.tensor(
        [
            [[1.0, 2.0], [10.0, 20.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        ]
    )
    mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])

    pooled = masked_mean(tokens, mask=mask)

    assert torch.equal(pooled[0], torch.tensor([2.0, 3.0]))
    assert torch.equal(pooled[1], torch.tensor([0.0, 0.0]))


def test_key_padding_mask_does_not_mask_all_tokens_for_all_missing_rows() -> None:
    mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])

    key_padding_mask = make_key_padding_mask(mask)

    assert torch.equal(key_padding_mask[0], torch.tensor([False, True, False]))
    assert torch.equal(key_padding_mask[1], torch.tensor([False, False, False]))


@pytest.mark.parametrize(
    "embedding_cls",
    [MaskedPoolingEmbedding, MaskedAttentionEmbedding],
)
def test_masked_embedding_forward_shape_and_finite(embedding_cls) -> None:
    torch.manual_seed(1)
    condition = torch.tensor(
        [
            [1.0, 0.0, 2.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    embedding = embedding_cls(x_dim=3, token_dim=8, context_dim=5, dropout=0.0)

    out = embedding(condition)

    assert out.shape == (2, 5)
    assert torch.isfinite(out).all()


def test_make_masked_embedding_factory() -> None:
    pooling = make_masked_embedding(
        MaskedEmbeddingConfig(
            embedding_type="masked_pooling",
            x_dim=4,
            token_dim=8,
            context_dim=6,
        )
    )
    attention = make_masked_embedding(
        MaskedEmbeddingConfig(
            embedding_type="masked_attention",
            x_dim=4,
            token_dim=8,
            context_dim=6,
            num_heads=2,
            num_layers=1,
        )
    )

    assert isinstance(pooling, MaskedPoolingEmbedding)
    assert isinstance(attention, MaskedAttentionEmbedding)
