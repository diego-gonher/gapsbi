import pytest

torch = pytest.importorskip("torch")

from gapsbi.methods.mask_augmentation import make_zero_imputed_mask_augmented_input


def test_mask_augmentation_shape_and_content() -> None:
    x_obs_scaled = torch.tensor(
        [[1.0, -2.0, 9.0], [3.0, 4.0, -1.0]],
        dtype=torch.float32,
    )
    mask = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.float32)

    x_aug = make_zero_imputed_mask_augmented_input(x_obs_scaled=x_obs_scaled, mask=mask)

    assert x_aug.shape == (2, 6)
    x_first = x_aug[:, :3]
    x_second = x_aug[:, 3:]

    expected_first = torch.tensor([[1.0, 0.0, 9.0], [0.0, 4.0, 0.0]], dtype=torch.float32)
    expected_second = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32)

    assert torch.equal(x_first, expected_first)
    assert torch.equal(x_second, expected_second)
    assert x_second.dtype == torch.float32


def test_mask_augmentation_observed_preserved_missing_zero() -> None:
    x_obs_scaled = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
    mask = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)

    x_aug = make_zero_imputed_mask_augmented_input(x_obs_scaled=x_obs_scaled, mask=mask)
    x_imputed = x_aug[:, :2]

    assert x_imputed[0, 0].item() == pytest.approx(5.0)
    assert x_imputed[1, 1].item() == pytest.approx(8.0)
    assert x_imputed[0, 1].item() == pytest.approx(0.0)
    assert x_imputed[1, 0].item() == pytest.approx(0.0)


def test_mask_not_scaled_and_inputs_not_mutated() -> None:
    x_obs_scaled = torch.tensor([[1.5, -3.0], [0.2, 0.0]], dtype=torch.float32)
    mask = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
    x_before = x_obs_scaled.clone()
    mask_before = mask.clone()

    x_aug = make_zero_imputed_mask_augmented_input(x_obs_scaled=x_obs_scaled, mask=mask)

    assert torch.equal(mask, mask_before)
    assert torch.equal(x_obs_scaled, x_before)
    assert torch.equal(x_aug[:, 2:], mask_before)
