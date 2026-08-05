from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gapsbi.methods.learned_imputation import _density_log_prob
from gapsbi.methods.sbi_npe import FixedSplitNPE_C


class LearnedConstantImputer(nn.Module):
    """Lueckmann-style per-feature learned constants in scaled x-space."""

    def __init__(self, x_dim: int, init_value: float = 0.0) -> None:
        super().__init__()
        if x_dim <= 0:
            raise ValueError("x_dim must be positive.")
        self.constants = nn.Parameter(torch.full((x_dim,), float(init_value)))

    def forward(self, x_obs_scaled: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _validate_shapes(x_obs_scaled, mask)
        constants = self.constants.to(dtype=x_obs_scaled.dtype, device=x_obs_scaled.device)
        return mask * x_obs_scaled + (1.0 - mask) * constants.unsqueeze(0)


@dataclass
class LearnedConstantImputationResult:
    inference: FixedSplitNPE_C
    imputer: LearnedConstantImputer
    density_estimator: nn.Module
    posterior: Any
    train_history: list[dict[str, float]]
    best_epoch: int
    best_val_loss: float
    stopped_epoch: int


def train_learned_constant_imputation_npe(
    *,
    theta_train: torch.Tensor,
    x_obs_train_scaled: torch.Tensor,
    mask_train: torch.Tensor,
    theta_val: torch.Tensor,
    x_obs_val_scaled: torch.Tensor,
    mask_val: torch.Tensor,
    prior,
    config: dict[str, Any],
) -> LearnedConstantImputationResult:
    """Jointly train per-feature imputation constants and an NPE estimator."""
    device = torch.device(str(config.get("device", "cpu")))
    batch_size = int(config.get("batch_size", config.get("training_batch_size", 256)))
    learning_rate = float(config.get("learning_rate", 1e-3))
    constant_learning_rate = float(config.get("constant_learning_rate", learning_rate))
    max_num_epochs = int(config.get("max_num_epochs", 5000))
    stop_after_epochs = int(config.get("stop_after_epochs", 20))
    density_estimator_name = str(config.get("density_estimator", "nsf"))
    init_value = float(config.get("init_value", 0.0))

    x_dim = x_obs_train_scaled.shape[1]
    imputer = LearnedConstantImputer(x_dim=x_dim, init_value=init_value).to(device)

    from sbi.neural_nets import posterior_nn

    density_builder = posterior_nn(model=density_estimator_name)
    density_estimator = density_builder(
        theta_train[: min(32, theta_train.shape[0])].to(device),
        x_obs_train_scaled[: min(32, x_obs_train_scaled.shape[0])].to(device),
    ).to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": density_estimator.parameters(), "lr": learning_rate},
            {"params": imputer.parameters(), "lr": constant_learning_rate},
        ]
    )

    train_loader = DataLoader(
        TensorDataset(theta_train, x_obs_train_scaled, mask_train),
        batch_size=min(batch_size, theta_train.shape[0]),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(theta_val, x_obs_val_scaled, mask_val),
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
        train_loss = 0.0
        train_batches = 0

        for theta_b, x_obs_b, mask_b in train_loader:
            theta_b = theta_b.to(device)
            x_obs_b = x_obs_b.to(device)
            mask_b = mask_b.to(device)

            optimizer.zero_grad(set_to_none=True)
            x_completed = imputer(x_obs_b, mask_b)
            loss = -_density_log_prob(
                density_estimator,
                theta_b,
                x_completed,
            ).mean()
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            train_batches += 1

        imputer.eval()
        density_estimator.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for theta_b, x_obs_b, mask_b in val_loader:
                theta_b = theta_b.to(device)
                x_obs_b = x_obs_b.to(device)
                mask_b = mask_b.to(device)

                x_completed = imputer(x_obs_b, mask_b)
                loss = -_density_log_prob(
                    density_estimator,
                    theta_b,
                    x_completed,
                ).mean()
                val_loss += float(loss.item())
                val_batches += 1

        train_loss /= max(train_batches, 1)
        val_loss /= max(val_batches, 1)
        train_history.append(
            {
                "epoch": float(epoch),
                "train_npe_loss": train_loss,
                "val_npe_loss": val_loss,
                "train_total_loss": train_loss,
                "val_total_loss": val_loss,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
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

    inference = FixedSplitNPE_C(
        prior=prior,
        density_estimator=density_estimator_name,
        device=str(device),
    )
    with torch.no_grad():
        x_completed_train = imputer(
            x_obs_train_scaled.to(device),
            mask_train.to(device),
        ).detach()
        x_completed_val = imputer(
            x_obs_val_scaled.to(device),
            mask_val.to(device),
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

    return LearnedConstantImputationResult(
        inference=inference,
        imputer=imputer,
        density_estimator=density_estimator,
        posterior=posterior,
        train_history=train_history,
        best_epoch=best_epoch,
        best_val_loss=float(best_val_loss),
        stopped_epoch=stopped_epoch,
    )


def _validate_shapes(x: torch.Tensor, mask: torch.Tensor) -> None:
    if x.ndim != 2:
        raise ValueError(f"x must have shape (N, D), got ndim={x.ndim}.")
    if mask.ndim != 2:
        raise ValueError(f"mask must have shape (N, D), got ndim={mask.ndim}.")
    if x.shape != mask.shape:
        raise ValueError(
            f"x and mask must match shapes, got {tuple(x.shape)} and {tuple(mask.shape)}."
        )
