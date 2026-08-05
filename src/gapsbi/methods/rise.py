from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from gapsbi.methods.sbi_npe import FixedSplitNPE_C
from gapsbi.utils.seeding import set_all_seeds


@dataclass
class RISEOutput:
    mean: torch.Tensor
    std: torch.Tensor
    completed_x: torch.Tensor
    mask_logits: Optional[torch.Tensor] = None
    latent_loc: Optional[torch.Tensor] = None
    latent_std: Optional[torch.Tensor] = None
    latent_sample: Optional[torch.Tensor] = None


@dataclass
class RISEResult:
    inference: FixedSplitNPE_C
    imputer: "RISEImputerMLP"
    density_estimator: nn.Module
    posterior: Any
    train_history: list[dict[str, float]]
    validation_history: list[dict[str, float]]
    final_train_loss: float
    final_validation_loss: float
    best_epoch: int
    best_val_loss: float
    stopped_epoch: int


class RISEImputerMLP(nn.Module):
    """MLP latent neural-process-style imputer for RISE-inspired training."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        latent_dim: int = 16,
        use_mask_head: bool = False,
        min_std: float = 1e-3,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if dropout < 0:
            raise ValueError("dropout must be nonnegative.")
        if min_std <= 0:
            raise ValueError("min_std must be positive.")

        self.input_dim = int(input_dim)
        self.use_mask_head = bool(use_mask_head)
        self.min_std = float(min_std)

        layers: list[nn.Module] = []
        current_dim = 2 * self.input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)

        self.latent_loc_head = nn.Linear(current_dim, latent_dim)
        self.latent_log_std_head = nn.Linear(current_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(current_dim + latent_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, self.input_dim)
        self.log_std_head = nn.Linear(hidden_dim, self.input_dim)
        self.mask_head = (
            nn.Linear(current_dim, self.input_dim) if self.use_mask_head else None
        )

    def forward(self, x_obs: torch.Tensor, mask: torch.Tensor) -> RISEOutput:
        _validate_x_mask(x_obs=x_obs, mask=mask, input_dim=self.input_dim)
        mask = mask.to(dtype=x_obs.dtype, device=x_obs.device)

        hidden = self.encoder(torch.cat([x_obs, mask], dim=-1))
        latent_loc = self.latent_loc_head(hidden)
        latent_std = F.softplus(self.latent_log_std_head(hidden)) + self.min_std
        latent_sample = latent_loc + latent_std * torch.randn_like(latent_std)
        decoded = self.decoder(torch.cat([hidden, latent_sample], dim=-1))
        mean = self.mean_head(decoded)
        std = F.softplus(self.log_std_head(decoded)) + self.min_std
        completed_x = mask * x_obs + (1.0 - mask) * mean
        mask_logits = self.mask_head(hidden) if self.mask_head is not None else None

        return RISEOutput(
            mean=mean,
            std=std,
            completed_x=completed_x,
            mask_logits=mask_logits,
            latent_loc=latent_loc,
            latent_std=latent_std,
            latent_sample=latent_sample,
        )


def train_rise_npe(
    *,
    theta_train: torch.Tensor,
    x_full_train: torch.Tensor,
    x_obs_train: torch.Tensor,
    mask_train: torch.Tensor,
    theta_val: torch.Tensor,
    x_full_val: torch.Tensor,
    x_obs_val: torch.Tensor,
    mask_val: torch.Tensor,
    prior,
    density_estimator: str = "nsf",
    device: str = "cpu",
    npe_training_batch_size: int = 256,
    npe_stop_after_epochs: int = 20,
    npe_max_num_epochs: int = 5000,
    learning_rate: float = 1e-3,
    rise_batch_size: int = 256,
    rise_max_num_epochs: int = 5000,
    rise_stop_after_epochs: int = 20,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.0,
    latent_dim: int = 16,
    lambda_np: float = 100.0,
    lambda_mask: float = 1.0,
    use_mask_head: bool = False,
    min_std: float = 1e-3,
    seed: Optional[int] = None,
) -> RISEResult:
    """Jointly train a RISE probabilistic imputer and NPE density estimator."""
    # The density estimator is optimized inside the joint RISE loop below.
    del npe_training_batch_size, npe_stop_after_epochs, npe_max_num_epochs

    if seed is not None:
        set_all_seeds(int(seed))

    _validate_training_arrays(
        theta_train=theta_train,
        x_full_train=x_full_train,
        x_obs_train=x_obs_train,
        mask_train=mask_train,
        theta_val=theta_val,
        x_full_val=x_full_val,
        x_obs_val=x_obs_val,
        mask_val=mask_val,
    )

    train_device = torch.device(str(device))
    x_dim = x_obs_train.shape[1]
    imputer = RISEImputerMLP(
        input_dim=x_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        latent_dim=latent_dim,
        use_mask_head=use_mask_head,
        min_std=min_std,
    ).to(train_device)

    # Lazy import: keep module importable even when sbi is absent.
    from sbi.neural_nets import posterior_nn

    density_builder = posterior_nn(model=density_estimator)
    density_net = density_builder(
        theta_train[: min(32, theta_train.shape[0])].to(train_device),
        x_obs_train[: min(32, x_obs_train.shape[0])].to(train_device),
    ).to(train_device)

    optimizer = torch.optim.Adam(
        list(imputer.parameters()) + list(density_net.parameters()),
        lr=float(learning_rate),
    )

    train_loader = DataLoader(
        TensorDataset(theta_train, x_full_train, x_obs_train, mask_train),
        batch_size=min(int(rise_batch_size), theta_train.shape[0]),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(theta_val, x_full_val, x_obs_val, mask_val),
        batch_size=min(int(rise_batch_size), theta_val.shape[0]),
        shuffle=False,
        drop_last=False,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_since_improvement = 0
    best_imputer_state = copy.deepcopy(imputer.state_dict())
    best_density_state = copy.deepcopy(density_net.state_dict())
    train_history: list[dict[str, float]] = []
    validation_history: list[dict[str, float]] = []

    for epoch in range(1, int(rise_max_num_epochs) + 1):
        imputer.train()
        density_net.train()
        train_metrics = _run_rise_epoch(
            data_loader=train_loader,
            imputer=imputer,
            density_estimator=density_net,
            optimizer=optimizer,
            device=train_device,
            lambda_np=lambda_np,
            lambda_mask=lambda_mask,
            use_mask_head=use_mask_head,
        )

        imputer.eval()
        density_net.eval()
        with torch.no_grad():
            val_metrics = _run_rise_epoch(
                data_loader=val_loader,
                imputer=imputer,
                density_estimator=density_net,
                optimizer=None,
                device=train_device,
                lambda_np=lambda_np,
                lambda_mask=lambda_mask,
                use_mask_head=use_mask_head,
            )

        train_history.append({"epoch": float(epoch), **train_metrics})
        validation_history.append({"epoch": float(epoch), **val_metrics})

        val_total = val_metrics["total_loss"]
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_epoch = epoch
            epochs_since_improvement = 0
            best_imputer_state = copy.deepcopy(imputer.state_dict())
            best_density_state = copy.deepcopy(density_net.state_dict())
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= int(rise_stop_after_epochs):
                break

    stopped_epoch = int(train_history[-1]["epoch"]) if train_history else 0
    imputer.load_state_dict(best_imputer_state)
    density_net.load_state_dict(best_density_state)
    imputer.eval()
    density_net.eval()

    with torch.no_grad():
        train_output = imputer(
            x_obs_train.to(train_device),
            mask_train.to(train_device),
        )
        val_output = imputer(
            x_obs_val.to(train_device),
            mask_val.to(train_device),
        )

    inference = FixedSplitNPE_C(
        prior=prior,
        density_estimator=density_estimator,
        device=str(train_device),
    )
    inference.append_simulations(
        torch.cat([theta_train, theta_val], dim=0).detach().cpu(),
        torch.cat(
            [
                train_output.completed_x.detach().cpu(),
                val_output.completed_x.detach().cpu(),
            ],
            dim=0,
        ),
    )
    inference.set_fixed_train_val_split(
        n_train=theta_train.shape[0],
        n_val=theta_val.shape[0],
    )
    posterior = inference.build_posterior(density_net)

    final_train_loss = (
        float(train_history[-1]["total_loss"]) if train_history else float("nan")
    )
    final_validation_loss = (
        float(validation_history[-1]["total_loss"])
        if validation_history
        else float("nan")
    )

    return RISEResult(
        inference=inference,
        imputer=imputer,
        density_estimator=density_net,
        posterior=posterior,
        train_history=train_history,
        validation_history=validation_history,
        final_train_loss=final_train_loss,
        final_validation_loss=final_validation_loss,
        best_epoch=best_epoch,
        best_val_loss=float(best_val_loss),
        stopped_epoch=stopped_epoch,
    )


def compute_rise_losses(
    *,
    x_full: torch.Tensor,
    x_obs: torch.Tensor,
    mask: torch.Tensor,
    output: RISEOutput,
    lambda_np: float = 1.0,
    lambda_mask: float = 1.0,
    use_mask_head: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute RISE auxiliary losses, excluding the NPE loss."""
    _validate_loss_inputs(x_full=x_full, x_obs=x_obs, mask=mask, output=output)
    if lambda_np < 0:
        raise ValueError("lambda_np must be nonnegative.")
    if lambda_mask < 0:
        raise ValueError("lambda_mask must be nonnegative.")

    distribution = torch.distributions.Normal(output.mean, output.std)
    np_loss = -distribution.log_prob(x_full).mean()

    if use_mask_head:
        if output.mask_logits is None:
            raise ValueError("output.mask_logits is required when use_mask_head=True.")
        mask_target = mask.to(
            dtype=output.mask_logits.dtype,
            device=output.mask_logits.device,
        )
        mask_loss = F.binary_cross_entropy_with_logits(output.mask_logits, mask_target)
    else:
        mask_loss = torch.zeros((), dtype=np_loss.dtype, device=np_loss.device)

    total_aux_loss = float(lambda_np) * np_loss + float(lambda_mask) * mask_loss
    return {
        "np_loss": np_loss,
        "mask_loss": mask_loss,
        "total_aux_loss": total_aux_loss,
    }


def _run_rise_epoch(
    *,
    data_loader: DataLoader,
    imputer: RISEImputerMLP,
    density_estimator: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    lambda_np: float,
    lambda_mask: float,
    use_mask_head: bool,
) -> dict[str, float]:
    total_loss_sum = 0.0
    npe_loss_sum = 0.0
    np_loss_sum = 0.0
    mask_loss_sum = 0.0
    aux_loss_sum = 0.0
    batches = 0

    for theta_b, x_full_b, x_obs_b, mask_b in data_loader:
        theta_b = theta_b.to(device)
        x_full_b = x_full_b.to(device)
        x_obs_b = x_obs_b.to(device)
        mask_b = mask_b.to(device)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        output = imputer(x_obs_b, mask_b)
        npe_loss = -_density_log_prob(
            density_estimator,
            theta_b,
            output.completed_x,
        ).mean()
        rise_losses = compute_rise_losses(
            x_full=x_full_b,
            x_obs=x_obs_b,
            mask=mask_b,
            output=output,
            lambda_np=lambda_np,
            lambda_mask=lambda_mask,
            use_mask_head=use_mask_head,
        )
        total_loss = npe_loss + rise_losses["total_aux_loss"]

        if optimizer is not None:
            total_loss.backward()
            optimizer.step()

        total_loss_sum += float(total_loss.item())
        npe_loss_sum += float(npe_loss.item())
        np_loss_sum += float(rise_losses["np_loss"].item())
        mask_loss_sum += float(rise_losses["mask_loss"].item())
        aux_loss_sum += float(rise_losses["total_aux_loss"].item())
        batches += 1

    denom = max(batches, 1)
    return {
        "total_loss": total_loss_sum / denom,
        "npe_loss": npe_loss_sum / denom,
        "np_loss": np_loss_sum / denom,
        "mask_loss": mask_loss_sum / denom,
        "aux_loss": aux_loss_sum / denom,
    }


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


def _validate_training_arrays(
    *,
    theta_train: torch.Tensor,
    x_full_train: torch.Tensor,
    x_obs_train: torch.Tensor,
    mask_train: torch.Tensor,
    theta_val: torch.Tensor,
    x_full_val: torch.Tensor,
    x_obs_val: torch.Tensor,
    mask_val: torch.Tensor,
) -> None:
    for name, theta in {"theta_train": theta_train, "theta_val": theta_val}.items():
        if not torch.is_tensor(theta):
            raise ValueError(f"{name} must be a torch.Tensor.")
        if theta.ndim != 2:
            raise ValueError(f"{name} must have shape (N, D), got ndim={theta.ndim}.")
        if theta.shape[0] <= 0:
            raise ValueError(f"{name} must contain at least one example.")

    _validate_split_arrays(
        split="train",
        theta=theta_train,
        x_full=x_full_train,
        x_obs=x_obs_train,
        mask=mask_train,
    )
    _validate_split_arrays(
        split="val",
        theta=theta_val,
        x_full=x_full_val,
        x_obs=x_obs_val,
        mask=mask_val,
    )
    if theta_train.shape[1] != theta_val.shape[1]:
        raise ValueError("theta_train and theta_val must have the same feature dim.")
    if x_obs_train.shape[1] != x_obs_val.shape[1]:
        raise ValueError("x_obs_train and x_obs_val must have the same feature dim.")


def _validate_split_arrays(
    *,
    split: str,
    theta: torch.Tensor,
    x_full: torch.Tensor,
    x_obs: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    _validate_x_mask(x_obs=x_obs, mask=mask, input_dim=x_obs.shape[1])
    if not torch.is_tensor(x_full):
        raise ValueError(f"x_full_{split} must be a torch.Tensor.")
    if x_full.shape != x_obs.shape:
        raise ValueError(
            f"x_full_{split} and x_obs_{split} must match shapes, got "
            f"{tuple(x_full.shape)} and {tuple(x_obs.shape)}."
        )
    if x_obs.shape[0] != theta.shape[0]:
        raise ValueError(
            f"x_obs_{split} rows must match theta_{split} rows, got "
            f"{x_obs.shape[0]} and {theta.shape[0]}."
        )


def _validate_x_mask(
    *,
    x_obs: torch.Tensor,
    mask: torch.Tensor,
    input_dim: int,
) -> None:
    if not torch.is_tensor(x_obs):
        raise ValueError("x_obs must be a torch.Tensor.")
    if not torch.is_tensor(mask):
        raise ValueError("mask must be a torch.Tensor.")
    if x_obs.ndim != 2:
        raise ValueError(f"x_obs must have shape (N, D), got ndim={x_obs.ndim}.")
    if mask.ndim != 2:
        raise ValueError(f"mask must have shape (N, D), got ndim={mask.ndim}.")
    if x_obs.shape != mask.shape:
        raise ValueError(
            f"x_obs and mask must have identical shapes, got "
            f"{tuple(x_obs.shape)} and {tuple(mask.shape)}."
        )
    if x_obs.shape[1] != input_dim:
        raise ValueError(
            f"x_obs feature dimension must match input_dim={input_dim}, "
            f"got {x_obs.shape[1]}."
        )


def _validate_loss_inputs(
    *,
    x_full: torch.Tensor,
    x_obs: torch.Tensor,
    mask: torch.Tensor,
    output: RISEOutput,
) -> None:
    if not torch.is_tensor(x_full):
        raise ValueError("x_full must be a torch.Tensor.")
    if not torch.is_tensor(x_obs):
        raise ValueError("x_obs must be a torch.Tensor.")
    if not torch.is_tensor(mask):
        raise ValueError("mask must be a torch.Tensor.")
    for name, tensor in {
        "output.mean": output.mean,
        "output.std": output.std,
        "output.completed_x": output.completed_x,
    }.items():
        if not torch.is_tensor(tensor):
            raise ValueError(f"{name} must be a torch.Tensor.")
        if tensor.shape != x_full.shape:
            raise ValueError(
                f"{name} must match x_full shape, got "
                f"{tuple(tensor.shape)} and {tuple(x_full.shape)}."
            )
    if mask.shape != x_full.shape:
        raise ValueError(
            f"mask must match x_full shape, got "
            f"{tuple(mask.shape)} and {tuple(x_full.shape)}."
        )
    if x_obs.shape != x_full.shape:
        raise ValueError(
            f"x_obs must match x_full shape, got "
            f"{tuple(x_obs.shape)} and {tuple(x_full.shape)}."
        )
    if torch.any(output.std <= 0):
        raise ValueError("output.std must be strictly positive.")
    if output.mask_logits is not None and output.mask_logits.shape != x_full.shape:
        raise ValueError(
            f"output.mask_logits must match x_full shape, got "
            f"{tuple(output.mask_logits.shape)} and {tuple(x_full.shape)}."
        )
