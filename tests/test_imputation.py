import pytest

torch = pytest.importorskip("torch")

from gapsbi.methods.imputation import (
    compute_observed_feature_means,
    mean_impute,
    zero_impute,
)


def test_zero_impute_preserves_observed_and_fills_missing_with_zero() -> None:
    x_obs_scaled = torch.tensor([[1.0, -2.0, 9.0], [3.0, 4.0, -1.0]])
    mask = torch.tensor([[1, 0, 1], [0, 1, 0]])

    imputed = zero_impute(x_obs_scaled=x_obs_scaled, mask=mask)

    expected = torch.tensor([[1.0, 0.0, 9.0], [0.0, 4.0, 0.0]])
    assert torch.equal(imputed, expected)


def test_mean_impute_preserves_observed_and_fills_missing_with_feature_means() -> None:
    x_obs_scaled = torch.tensor([[1.0, -2.0, 9.0], [3.0, 4.0, -1.0]])
    mask = torch.tensor([[1, 0, 1], [0, 1, 0]])
    feature_means = torch.tensor([10.0, 20.0, 30.0])

    imputed = mean_impute(x_obs_scaled=x_obs_scaled, mask=mask, feature_means=feature_means)

    expected = torch.tensor([[1.0, 20.0, 9.0], [10.0, 4.0, 30.0]])
    assert torch.equal(imputed, expected)


def test_compute_observed_feature_means_uses_only_observed_entries() -> None:
    x_obs_scaled_train = torch.tensor(
        [
            [1.0, -99.0, 5.0],
            [3.0, 4.0, -99.0],
            [5.0, 8.0, 7.0],
        ]
    )
    mask_train = torch.tensor(
        [
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )

    means = compute_observed_feature_means(
        x_obs_scaled_train=x_obs_scaled_train,
        mask_train=mask_train,
    )

    # col0: (1 + 3 + 5) / 3 = 3
    # col1: (4 + 8) / 2 = 6
    # col2: (5 + 7) / 2 = 6
    expected = torch.tensor([3.0, 6.0, 6.0])
    assert torch.allclose(means, expected)


def test_compute_observed_feature_means_raises_for_fully_missing_feature() -> None:
    x_obs_scaled_train = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask_train = torch.tensor([[1, 0], [1, 0]])

    with pytest.raises(ValueError, match="zero observed entries"):
        compute_observed_feature_means(
            x_obs_scaled_train=x_obs_scaled_train,
            mask_train=mask_train,
        )


def test_imputation_functions_do_not_mutate_inputs() -> None:
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask = torch.tensor([[1, 0], [0, 1]])
    means = torch.tensor([10.0, 20.0])

    x_before = x.clone()
    mask_before = mask.clone()
    means_before = means.clone()

    _ = zero_impute(x_obs_scaled=x, mask=mask)
    _ = mean_impute(x_obs_scaled=x, mask=mask, feature_means=means)
    _ = compute_observed_feature_means(x_obs_scaled_train=x, mask_train=mask)

    assert torch.equal(x, x_before)
    assert torch.equal(mask, mask_before)
    assert torch.equal(means, means_before)


def test_shape_mismatch_raises_clear_value_error() -> None:
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    bad_mask = torch.tensor([[1, 0, 1], [0, 1, 0]])
    means = torch.tensor([10.0, 20.0])

    with pytest.raises(ValueError, match="identical shapes"):
        zero_impute(x_obs_scaled=x, mask=bad_mask)

    with pytest.raises(ValueError, match="identical shapes"):
        compute_observed_feature_means(x_obs_scaled_train=x, mask_train=bad_mask)

    with pytest.raises(ValueError, match="length D=2"):
        mean_impute(
            x_obs_scaled=x,
            mask=torch.tensor([[1, 0], [0, 1]]),
            feature_means=torch.tensor([1.0, 2.0, 3.0]),
        )
