import pytest

torch = pytest.importorskip("torch")
sbi = pytest.importorskip("sbi")
from sbi.utils import BoxUniform

from gapsbi.methods.rise import RISEImputerMLP, compute_rise_losses, train_rise_npe


def test_rise_imputer_output_shapes() -> None:
    model = RISEImputerMLP(input_dim=5, hidden_dim=16, num_layers=2, latent_dim=4)
    x_obs = torch.randn(3, 5)
    mask = torch.randint(0, 2, (3, 5)).float()

    output = model(x_obs, mask)

    assert output.mean.shape == (3, 5)
    assert output.std.shape == (3, 5)
    assert output.completed_x.shape == (3, 5)
    assert output.mask_logits is None
    assert output.latent_loc is not None
    assert output.latent_std is not None
    assert output.latent_sample is not None
    assert output.latent_loc.shape == (3, 4)
    assert output.latent_sample.shape == (3, 4)


def test_rise_imputer_std_is_strictly_positive() -> None:
    model = RISEImputerMLP(input_dim=4, hidden_dim=8, min_std=1e-3)
    x_obs = torch.randn(2, 4)
    mask = torch.ones(2, 4)

    output = model(x_obs, mask)

    assert torch.all(output.std > 0)


def test_rise_completion_keeps_observed_entries() -> None:
    model = RISEImputerMLP(input_dim=3, hidden_dim=8)
    x_obs = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0]])

    output = model(x_obs, mask)

    assert output.completed_x[0, 0].item() == pytest.approx(1.0)
    assert output.completed_x[0, 2].item() == pytest.approx(3.0)
    assert output.completed_x[0, 1].item() == pytest.approx(output.mean[0, 1].item())


def test_rise_loss_computation_returns_finite_values() -> None:
    model = RISEImputerMLP(input_dim=4, hidden_dim=8)
    x_full = torch.randn(6, 4)
    mask = (torch.rand(6, 4) > 0.4).float()
    x_obs = x_full * mask
    output = model(x_obs, mask)

    losses = compute_rise_losses(
        x_full=x_full,
        x_obs=x_obs,
        mask=mask,
        output=output,
    )

    assert set(losses) >= {"np_loss", "mask_loss", "total_aux_loss"}
    assert torch.isfinite(losses["np_loss"])
    assert torch.isfinite(losses["mask_loss"])
    assert torch.isfinite(losses["total_aux_loss"])


def test_rise_mnar_mode_returns_mask_logits_and_loss() -> None:
    model = RISEImputerMLP(
        input_dim=4,
        hidden_dim=8,
        latent_dim=4,
        use_mask_head=True,
    )
    x_full = torch.randn(5, 4)
    mask = (torch.rand(5, 4) > 0.5).float()
    x_obs = x_full * mask

    output = model(x_obs, mask)
    losses = compute_rise_losses(
        x_full=x_full,
        x_obs=x_obs,
        mask=mask,
        output=output,
        use_mask_head=True,
    )

    assert output.mask_logits is not None
    assert output.mask_logits.shape == (5, 4)
    assert torch.isfinite(losses["mask_loss"])


def test_train_rise_npe_runs_tiny_dataset() -> None:
    theta_train = torch.randn(16, 2)
    theta_val = torch.randn(8, 2)
    x_full_train = torch.randn(16, 6)
    x_full_val = torch.randn(8, 6)
    mask_train = (torch.rand(16, 6) > 0.3).float()
    mask_val = (torch.rand(8, 6) > 0.3).float()
    x_obs_train = x_full_train * mask_train
    x_obs_val = x_full_val * mask_val
    prior = BoxUniform(low=-torch.ones(2), high=torch.ones(2))

    result = train_rise_npe(
        theta_train=theta_train,
        x_full_train=x_full_train,
        x_obs_train=x_obs_train,
        mask_train=mask_train,
        theta_val=theta_val,
        x_full_val=x_full_val,
        x_obs_val=x_obs_val,
        mask_val=mask_val,
        prior=prior,
        density_estimator="nsf",
        device="cpu",
        rise_batch_size=8,
        rise_max_num_epochs=1,
        rise_stop_after_epochs=1,
        hidden_dim=8,
        num_layers=1,
        latent_dim=4,
        seed=123,
    )

    assert isinstance(result.imputer, RISEImputerMLP)
    assert result.validation_history
    assert torch.isfinite(torch.tensor(result.final_train_loss))
    assert torch.isfinite(torch.tensor(result.final_validation_loss))
