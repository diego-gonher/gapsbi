from __future__ import annotations

import copy
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gapsbi.methods.sbi_npe import FixedSplitNPE_C
from gapsbi.preprocessing.scalers import (
    fit_x_scaler_on_full_train,
    infer_theta_transform,
    infer_x_transform,
    scale_theta_train_val_test,
    theta_scaling_metadata,
    transform_x_obs_with_fitted_scaler,
)


class MLPImputer(nn.Module):
    """Small MLP imputer over concatenated [x_obs, mask]."""

    def __init__(
        self,
        x_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if x_dim <= 0:
            raise ValueError("x_dim must be positive.")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")

        in_dim = 2 * x_dim
        layers: list[nn.Module] = []
        current = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current = hidden_dim
        layers.append(nn.Linear(current, x_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_obs_scaled: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _validate_shapes(x_obs_scaled, mask)
        x_in = torch.cat([x_obs_scaled, mask], dim=-1)
        return self.net(x_in)


class CNN1DImputer(nn.Module):
    """Lightweight 1D CNN imputer using channels [x_obs, mask]."""

    def __init__(
        self,
        x_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if x_dim <= 0:
            raise ValueError("x_dim must be positive.")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")

        blocks: list[nn.Module] = []
        in_channels = 2
        for _ in range(num_layers):
            blocks.append(nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1))
            blocks.append(nn.ReLU())
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
            in_channels = hidden_dim
        blocks.append(nn.Conv1d(in_channels, 1, kernel_size=3, padding=1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x_obs_scaled: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _validate_shapes(x_obs_scaled, mask)
        x_in = torch.stack([x_obs_scaled, mask], dim=1)  # (N, 2, D)
        x_hat = self.net(x_in)[:, 0, :]  # (N, D)
        return x_hat


def build_imputer(problem: str, x_dim: int, config: dict[str, Any]) -> nn.Module:
    """Build imputer from config with auto heuristic by problem."""
    imputer_type = str(config.get("imputer_type", "auto")).lower()
    hidden_dim = int(config.get("hidden_dim", 128))
    num_layers = int(config.get("num_layers", 2))
    dropout = float(config.get("dropout", 0.0))

    if imputer_type == "auto":
        resolved = "cnn" if problem in {"oup", "lotka_volterra", "ricker"} else "mlp"
    else:
        resolved = imputer_type

    if resolved == "mlp":
        return MLPImputer(
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    if resolved == "cnn":
        return CNN1DImputer(
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    raise ValueError(
        "imputer_type must be one of {'auto', 'mlp', 'cnn'}, "
        f"got {imputer_type!r}."
    )


def prepare_learned_imputation_arrays(
    dataset: dict[str, dict[str, np.ndarray]],
    problem: str,
) -> dict[str, Any]:
    """Prepare theta/x/mask tensors with train-only fitted scalers."""
    theta_train_np = dataset["train"]["theta"]
    theta_val_np = dataset["val"]["theta"]
    theta_test_np = dataset["test"]["theta"]

    x_full_train_np = dataset["train"]["x_full"]
    x_full_val_np = dataset["val"]["x_full"]
    x_full_test_np = dataset["test"]["x_full"]

    x_obs_train_np = dataset["train"]["x_obs"]
    x_obs_val_np = dataset["val"]["x_obs"]
    x_obs_test_np = dataset["test"]["x_obs"]

    mask_train_np = dataset["train"]["mask"]
    mask_val_np = dataset["val"]["mask"]
    mask_test_np = dataset["test"]["mask"]

    theta_transform = infer_theta_transform(problem)
    theta_train, theta_val, theta_test, theta_scaler = scale_theta_train_val_test(
        theta_train=theta_train_np,
        theta_val=theta_val_np,
        theta_test=theta_test_np,
        transform=theta_transform,
    )

    x_transform = infer_x_transform(problem)
    x_scaler, x_scaling_metadata = fit_x_scaler_on_full_train(
        x_full_train=x_full_train_np,
        transform=x_transform,
    )
    x_full_train_scaled = _transform_x_full_with_fitted_scaler(
        x_full=x_full_train_np,
        x_scaler=x_scaler,
        transform=x_transform,
    )
    x_full_val_scaled = _transform_x_full_with_fitted_scaler(
        x_full=x_full_val_np,
        x_scaler=x_scaler,
        transform=x_transform,
    )
    x_full_test_scaled = _transform_x_full_with_fitted_scaler(
        x_full=x_full_test_np,
        x_scaler=x_scaler,
        transform=x_transform,
    )

    x_obs_train_scaled = transform_x_obs_with_fitted_scaler(
        x_obs=x_obs_train_np,
        mask=mask_train_np,
        x_scaler=x_scaler,
        transform=x_transform,
    )
    x_obs_val_scaled = transform_x_obs_with_fitted_scaler(
        x_obs=x_obs_val_np,
        mask=mask_val_np,
        x_scaler=x_scaler,
        transform=x_transform,
    )
    x_obs_test_scaled = transform_x_obs_with_fitted_scaler(
        x_obs=x_obs_test_np,
        mask=mask_test_np,
        x_scaler=x_scaler,
        transform=x_transform,
    )

    return {
        "theta_train": theta_train,
        "theta_val": theta_val,
        "theta_test": theta_test,
        "x_full_train_scaled": x_full_train_scaled,
        "x_full_val_scaled": x_full_val_scaled,
        "x_full_test_scaled": x_full_test_scaled,
        "x_obs_train_scaled": x_obs_train_scaled,
        "x_obs_val_scaled": x_obs_val_scaled,
        "x_obs_test_scaled": x_obs_test_scaled,
        "mask_train": torch.tensor(mask_train_np, dtype=torch.float32),
        "mask_val": torch.tensor(mask_val_np, dtype=torch.float32),
        "mask_test": torch.tensor(mask_test_np, dtype=torch.float32),
        "theta_scaler": theta_scaler,
        "theta_transform": theta_transform,
        "theta_scaling_metadata": theta_scaling_metadata(theta_transform),
        "x_scaler": x_scaler,
        "x_transform": x_transform,
        "x_scaling_metadata": x_scaling_metadata,
    }


@dataclass
class LearnedImputationResult:
    inference: FixedSplitNPE_C
    imputer: nn.Module
    density_estimator: nn.Module
    posterior: Any
    train_history: list[dict[str, float]]
    best_epoch: int
    best_val_loss: float
    stopped_epoch: int


def train_learned_imputation_npe(
    *,
    problem: str,
    theta_train: torch.Tensor,
    x_obs_train_scaled: torch.Tensor,
    x_full_train_scaled: torch.Tensor,
    mask_train: torch.Tensor,
    theta_val: torch.Tensor,
    x_obs_val_scaled: torch.Tensor,
    x_full_val_scaled: torch.Tensor,
    mask_val: torch.Tensor,
    prior,
    config: dict[str, Any],
) -> LearnedImputationResult:
    """Jointly train deterministic imputer + NPE estimator."""
    device = torch.device(str(config.get("device", "cpu")))
    batch_size = int(config.get("batch_size", config.get("training_batch_size", 256)))
    learning_rate = float(config.get("learning_rate", 1e-3))
    max_num_epochs = int(config.get("max_num_epochs", 5000))
    stop_after_epochs = int(config.get("stop_after_epochs", 20))
    lambda_recon = float(config.get("lambda_recon", 1.0))
    density_estimator_name = str(config.get("density_estimator", "nsf"))

    x_dim = x_obs_train_scaled.shape[1]
    imputer = build_imputer(problem=problem, x_dim=x_dim, config=config).to(device)

    # Lazy import: keep module importable even when sbi is absent.
    from sbi.neural_nets import posterior_nn

    density_builder = posterior_nn(model=density_estimator_name)
    density_estimator = density_builder(
        theta_train[: min(32, theta_train.shape[0])].to(device),
        x_obs_train_scaled[: min(32, x_obs_train_scaled.shape[0])].to(device),
    ).to(device)

    optimizer = torch.optim.Adam(
        list(imputer.parameters()) + list(density_estimator.parameters()),
        lr=learning_rate,
    )

    train_loader = DataLoader(
        TensorDataset(theta_train, x_obs_train_scaled, x_full_train_scaled, mask_train),
        batch_size=min(batch_size, theta_train.shape[0]),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(theta_val, x_obs_val_scaled, x_full_val_scaled, mask_val),
        batch_size=min(batch_size, theta_val.shape[0]),
        shuffle=False,
        drop_last=False,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_since_improvement = 0
    best_imputer_state = copy.deepcopy(imputer.state_dict())
    best_density_state = copy.deepcopy(density_estimator.state_dict())
    train_history: list[dict[str, float]] = []

    for epoch in range(1, max_num_epochs + 1):
        imputer.train()
        density_estimator.train()
        train_total = 0.0
        train_npe = 0.0
        train_recon = 0.0
        train_batches = 0

        for theta_b, x_obs_b, x_full_b, mask_b in train_loader:
            theta_b = theta_b.to(device)
            x_obs_b = x_obs_b.to(device)
            x_full_b = x_full_b.to(device)
            mask_b = mask_b.to(device)

            optimizer.zero_grad(set_to_none=True)
            x_hat = imputer(x_obs_b, mask_b)
            x_completed = complete_with_imputer(x_obs_b, mask_b, x_hat)

            npe_loss = -_density_log_prob(
                density_estimator,
                theta_b,
                x_completed,
            ).mean()
            recon_loss = _missing_only_mse(x_hat, x_full_b, mask_b)
            total_loss = npe_loss + lambda_recon * recon_loss
            total_loss.backward()
            optimizer.step()

            train_total += float(total_loss.item())
            train_npe += float(npe_loss.item())
            train_recon += float(recon_loss.item())
            train_batches += 1

        imputer.eval()
        density_estimator.eval()
        val_total = 0.0
        val_npe = 0.0
        val_recon = 0.0
        val_batches = 0
        with torch.no_grad():
            for theta_b, x_obs_b, x_full_b, mask_b in val_loader:
                theta_b = theta_b.to(device)
                x_obs_b = x_obs_b.to(device)
                x_full_b = x_full_b.to(device)
                mask_b = mask_b.to(device)

                x_hat = imputer(x_obs_b, mask_b)
                x_completed = complete_with_imputer(x_obs_b, mask_b, x_hat)
                npe_loss = -_density_log_prob(
                    density_estimator,
                    theta_b,
                    x_completed,
                ).mean()
                recon_loss = _missing_only_mse(x_hat, x_full_b, mask_b)
                total_loss = npe_loss + lambda_recon * recon_loss

                val_total += float(total_loss.item())
                val_npe += float(npe_loss.item())
                val_recon += float(recon_loss.item())
                val_batches += 1

        train_total /= max(train_batches, 1)
        train_npe /= max(train_batches, 1)
        train_recon /= max(train_batches, 1)
        val_total /= max(val_batches, 1)
        val_npe /= max(val_batches, 1)
        val_recon /= max(val_batches, 1)

        train_history.append(
            {
                "epoch": float(epoch),
                "train_total_loss": train_total,
                "train_npe_loss": train_npe,
                "train_recon_loss": train_recon,
                "val_total_loss": val_total,
                "val_npe_loss": val_npe,
                "val_recon_loss": val_recon,
            }
        )

        if val_total < best_val_loss:
            best_val_loss = val_total
            best_epoch = epoch
            epochs_since_improvement = 0
            best_imputer_state = copy.deepcopy(imputer.state_dict())
            best_density_state = copy.deepcopy(density_estimator.state_dict())
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= stop_after_epochs:
                break

    stopped_epoch = int(train_history[-1]["epoch"]) if train_history else 0
    imputer.load_state_dict(best_imputer_state)
    density_estimator.load_state_dict(best_density_state)
    imputer.eval()
    density_estimator.eval()

    # Build sbi posterior from trained density estimator.
    inference = FixedSplitNPE_C(
        prior=prior,
        density_estimator=density_estimator_name,
        device=str(device),
    )
    with torch.no_grad():
        x_hat_train = imputer(
            x_obs_train_scaled.to(device),
            mask_train.to(device),
        ).detach()
        x_completed_train = complete_with_imputer(
            x_obs_train_scaled.to(device),
            mask_train.to(device),
            x_hat_train,
        ).detach()
        x_hat_val = imputer(
            x_obs_val_scaled.to(device),
            mask_val.to(device),
        ).detach()
        x_completed_val = complete_with_imputer(
            x_obs_val_scaled.to(device),
            mask_val.to(device),
            x_hat_val,
        ).detach()

    inference.append_simulations(
        torch.cat([theta_train, theta_val], dim=0).detach().cpu(),
        torch.cat([x_completed_train, x_completed_val], dim=0).detach().cpu(),
    )
    inference.set_fixed_train_val_split(
        n_train=theta_train.shape[0],
        n_val=theta_val.shape[0],
    )
    posterior = inference.build_posterior(density_estimator)

    return LearnedImputationResult(
        inference=inference,
        imputer=imputer,
        density_estimator=density_estimator,
        posterior=posterior,
        train_history=train_history,
        best_epoch=best_epoch,
        best_val_loss=float(best_val_loss),
        stopped_epoch=stopped_epoch,
    )


def complete_with_imputer(
    x_obs_scaled: torch.Tensor,
    mask: torch.Tensor,
    x_hat_full_scaled: torch.Tensor,
) -> torch.Tensor:
    _validate_shapes(x_obs_scaled, mask)
    if x_hat_full_scaled.shape != x_obs_scaled.shape:
        raise ValueError(
            f"x_hat_full_scaled must match x_obs_scaled shape, got "
            f"{tuple(x_hat_full_scaled.shape)} and {tuple(x_obs_scaled.shape)}."
        )
    return mask * x_obs_scaled + (1.0 - mask) * x_hat_full_scaled


def _missing_only_mse(
    x_hat_full_scaled: torch.Tensor,
    x_full_scaled: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    _validate_shapes(x_hat_full_scaled, mask)
    if x_full_scaled.shape != x_hat_full_scaled.shape:
        raise ValueError(
            f"x_full_scaled must match x_hat_full_scaled shape, got "
            f"{tuple(x_full_scaled.shape)} and {tuple(x_hat_full_scaled.shape)}."
        )
    missing = 1.0 - mask
    missing_count = missing.sum()
    if float(missing_count.item()) == 0.0:
        return torch.zeros((), dtype=x_hat_full_scaled.dtype, device=x_hat_full_scaled.device)
    squared_error = (missing * (x_hat_full_scaled - x_full_scaled)) ** 2
    return squared_error.sum() / missing_count


def _density_log_prob(
    density_estimator: nn.Module,
    theta: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    try:
        return density_estimator.log_prob(theta, context=context)
    except TypeError:
        try:
            return density_estimator.log_prob(theta, condition=context)
        except TypeError:
            return density_estimator.log_prob(theta, context)


def _transform_x_full_with_fitted_scaler(
    x_full: np.ndarray,
    x_scaler,
    transform: str,
) -> torch.Tensor:
    x_full_np = np.asarray(x_full, dtype=np.float64)
    if transform == "standard":
        features = x_full_np
    elif transform == "log1p_standard":
        if np.any(x_full_np < 0):
            raise ValueError("log1p_standard requires nonnegative x_full values.")
        features = np.log1p(x_full_np)
    else:
        raise ValueError(
            "Invalid transform. Expected one of {'standard', 'log1p_standard'}, "
            f"got {transform!r}."
        )
    return torch.tensor(x_scaler.transform(features), dtype=torch.float32)


def _validate_shapes(x: torch.Tensor, mask: torch.Tensor) -> None:
    if x.ndim != 2:
        raise ValueError(f"x must have shape (N, D), got ndim={x.ndim}.")
    if mask.ndim != 2:
        raise ValueError(f"mask must have shape (N, D), got ndim={mask.ndim}.")
    if x.shape != mask.shape:
        raise ValueError(
            f"x and mask must match shapes, got {tuple(x.shape)} and {tuple(mask.shape)}."
        )


def parse_missingness_and_epsilon(dataset_path: str | Path) -> tuple[str, str]:
    path = str(dataset_path).lower()
    mechanism = "unknown"
    for candidate in ("mcar", "mar", "mnar"):
        if f"/{candidate}/" in path:
            mechanism = candidate
            break

    match = re.search(r"eps(\d{3})", path)
    epsilon = match.group(1) if match else "unknown"
    return mechanism, epsilon
