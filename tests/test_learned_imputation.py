import pytest

torch = pytest.importorskip("torch")
sbi = pytest.importorskip("sbi")
from sbi.utils import BoxUniform

from gapsbi.methods.learned_imputation import (
    CNN1DImputer,
    MLPImputer,
    _missing_only_mse,
    complete_with_imputer,
    train_learned_imputation_npe,
)


def test_mlp_imputer_output_shape() -> None:
    model = MLPImputer(x_dim=5, hidden_dim=16, num_layers=2)
    x_obs = torch.randn(4, 5)
    mask = torch.randint(0, 2, (4, 5)).float()

    x_hat = model(x_obs, mask)
    assert x_hat.shape == (4, 5)


def test_cnn_imputer_output_shape() -> None:
    model = CNN1DImputer(x_dim=8, hidden_dim=8, num_layers=2)
    x_obs = torch.randn(3, 8)
    mask = torch.randint(0, 2, (3, 8)).float()

    x_hat = model(x_obs, mask)
    assert x_hat.shape == (3, 8)


def test_x_completed_keeps_observed_entries() -> None:
    x_obs = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    x_hat = torch.tensor([[9.0, 8.0, 7.0]])

    x_completed = complete_with_imputer(x_obs, mask, x_hat)
    assert x_completed[0, 0].item() == pytest.approx(1.0)
    assert x_completed[0, 2].item() == pytest.approx(3.0)
    assert x_completed[0, 1].item() == pytest.approx(8.0)


def test_reconstruction_loss_only_uses_missing_entries() -> None:
    x_hat = torch.tensor([[10.0, 20.0]])
    x_full = torch.tensor([[1.0, 2.0]])
    mask = torch.tensor([[1.0, 0.0]])

    loss = _missing_only_mse(x_hat, x_full, mask)
    # Only second entry should contribute: (20 - 2)^2 = 324
    assert loss.item() == pytest.approx(324.0)


def test_reconstruction_loss_no_crash_all_observed() -> None:
    x_hat = torch.tensor([[1.0, 2.0]])
    x_full = torch.tensor([[3.0, 4.0]])
    mask = torch.tensor([[1.0, 1.0]])

    loss = _missing_only_mse(x_hat, x_full, mask)
    assert loss.item() == pytest.approx(0.0)


def test_train_loop_runs_tiny_dataset() -> None:
    theta_train = torch.randn(16, 2)
    theta_val = torch.randn(8, 2)
    x_full_train = torch.randn(16, 10)
    x_full_val = torch.randn(8, 10)
    mask_train = (torch.rand(16, 10) > 0.3).float()
    mask_val = (torch.rand(8, 10) > 0.3).float()
    x_obs_train = x_full_train * mask_train
    x_obs_val = x_full_val * mask_val

    prior = BoxUniform(low=-torch.ones(2), high=torch.ones(2))

    result = train_learned_imputation_npe(
        problem="glu",
        theta_train=theta_train,
        x_obs_train_scaled=x_obs_train,
        x_full_train_scaled=x_full_train,
        mask_train=mask_train,
        theta_val=theta_val,
        x_obs_val_scaled=x_obs_val,
        x_full_val_scaled=x_full_val,
        mask_val=mask_val,
        prior=prior,
        config={
            "device": "cpu",
            "density_estimator": "nsf",
            "batch_size": 8,
            "learning_rate": 1e-3,
            "max_num_epochs": 2,
            "stop_after_epochs": 2,
            "lambda_recon": 1.0,
            "imputer_type": "mlp",
            "hidden_dim": 16,
            "num_layers": 2,
            "dropout": 0.0,
        },
    )

    assert result.best_epoch >= 1
    assert result.best_val_loss == pytest.approx(result.best_val_loss)
