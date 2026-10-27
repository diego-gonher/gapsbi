"""Small fixed-split wrapper for SBI's standard NRE-A trainer."""
from __future__ import annotations

import torch
import numpy as np
import copy
from dataclasses import dataclass
from sbi.inference import NRE_A, NRE_C
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler


def prepare_zero_imputed_nre_input(*, x_obs, mask, x_scaler, transform):
    """Apply the canonical scaled-space zero-imputation used by NPE.

    NRE-C receives this 25-dimensional vector directly: the mask is deliberately
    not concatenated to its classifier input.
    """
    from gapsbi.methods.imputation import zero_impute
    from gapsbi.preprocessing.scalers import transform_x_obs_with_fitted_scaler

    scaled = transform_x_obs_with_fitted_scaler(
        x_obs=np.asarray(x_obs), mask=np.asarray(mask), x_scaler=x_scaler,
        transform=str(transform),
    )
    return zero_impute(scaled, torch.as_tensor(mask, dtype=torch.float32))


def prepare_mask_augmented_nre_input(*, x_obs, mask, x_scaler, transform):
    """Apply the canonical NPE mask-augmentation representation to raw input."""
    from gapsbi.methods.mask_augmentation import make_zero_imputed_mask_augmented_input
    from gapsbi.preprocessing.scalers import transform_x_obs_with_fitted_scaler

    scaled = transform_x_obs_with_fitted_scaler(
        x_obs=np.asarray(x_obs), mask=np.asarray(mask), x_scaler=x_scaler,
        transform=str(transform),
    )
    return make_zero_imputed_mask_augmented_input(
        x_obs_scaled=scaled, mask=torch.as_tensor(mask, dtype=torch.float32)
    )


def load_nre_a_full_data_checkpoint(checkpoint_path, *, repository_root="."):
    """Rebuild an experimental full-data NRE-A ratio estimator for validation."""
    from pathlib import Path
    from gapsbi.datasets import apply_train_val_sample_limits
    from gapsbi.io import load_gapsbi_hdf5
    from gapsbi.preprocessing.scalers import make_scaled_theta_prior, scale_x_train_val_test

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    root = Path(repository_root)
    dataset_path = Path(checkpoint["dataset_path"])
    dataset, _ = load_gapsbi_hdf5(dataset_path if dataset_path.is_absolute() else root / dataset_path)
    config = checkpoint["config"]
    dataset = apply_train_val_sample_limits(
        dataset, config.get("max_train_samples"), config.get("max_val_samples")
    )
    theta_scaler = checkpoint["theta_scaler"]
    theta_train = torch.tensor(theta_scaler.transform(dataset["train"]["theta"]), dtype=torch.float32)
    theta_val = torch.tensor(theta_scaler.transform(dataset["val"]["theta"]), dtype=torch.float32)
    x_scaler = checkpoint["x_scaler"]
    x_train = torch.tensor(x_scaler.transform(dataset["train"]["x_full"]), dtype=torch.float32)
    x_val = torch.tensor(x_scaler.transform(dataset["val"]["x_full"]), dtype=torch.float32)
    prior = make_scaled_theta_prior(checkpoint["problem"], checkpoint["theta_dim"], theta_train)
    inference = FixedSplitNRE_A(prior=prior, classifier=config["nre"]["classifier"], device="cpu", show_progress_bars=False)
    inference.append_simulations(torch.cat((theta_train, theta_val)), torch.cat((x_train, x_val)))
    inference.set_fixed_train_val_split(len(theta_train), len(theta_val))
    inference._initialize_neural_network(retrain_from_scratch=False, start_idx=0)
    inference._neural_net.load_state_dict(checkpoint["density_estimator_state_dict"])
    inference._neural_net.eval()
    return checkpoint, inference._neural_net


def load_nre_c_full_data_checkpoint(checkpoint_path, *, repository_root="."):
    """Rebuild an experimental full-data NRE-C ratio estimator for validation."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    from pathlib import Path
    from gapsbi.datasets import apply_train_val_sample_limits
    from gapsbi.io import load_gapsbi_hdf5
    from gapsbi.preprocessing.scalers import make_scaled_theta_prior
    root, config = Path(repository_root), checkpoint["config"]
    path = Path(checkpoint["dataset_path"])
    dataset, _ = load_gapsbi_hdf5(path if path.is_absolute() else root / path)
    dataset = apply_train_val_sample_limits(
        dataset, config.get("max_train_samples"), config.get("max_val_samples")
    )
    theta_scaler, x_scaler = checkpoint["theta_scaler"], checkpoint["x_scaler"]
    theta_train = torch.tensor(theta_scaler.transform(dataset["train"]["theta"]), dtype=torch.float32)
    theta_val = torch.tensor(theta_scaler.transform(dataset["val"]["theta"]), dtype=torch.float32)
    x_train = torch.tensor(x_scaler.transform(dataset["train"]["x_full"]), dtype=torch.float32)
    x_val = torch.tensor(x_scaler.transform(dataset["val"]["x_full"]), dtype=torch.float32)
    inference = FixedSplitNRE_C(prior=make_scaled_theta_prior("oup", 2, theta_train), classifier=config["nre"]["classifier"], device="cpu", show_progress_bars=False)
    inference.append_simulations(torch.cat((theta_train, theta_val)), torch.cat((x_train, x_val)))
    inference.set_fixed_train_val_split(len(theta_train), len(theta_val))
    inference._initialize_neural_network(retrain_from_scratch=False, start_idx=0)
    inference._neural_net.load_state_dict(checkpoint["density_estimator_state_dict"])
    inference._neural_net.eval()
    return checkpoint, inference._neural_net


def load_nre_c_zero_imputation_checkpoint(checkpoint_path, *, repository_root="."):
    """Rebuild an experimental NRE-C estimator trained on zero-imputed x_obs."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    from pathlib import Path
    from gapsbi.datasets import apply_train_val_sample_limits
    from gapsbi.io import load_gapsbi_hdf5
    from gapsbi.preprocessing.scalers import make_scaled_theta_prior

    root, config = Path(repository_root), checkpoint["config"]
    dataset_path = Path(checkpoint["dataset_path"])
    dataset, _ = load_gapsbi_hdf5(
        dataset_path if dataset_path.is_absolute() else root / dataset_path
    )
    dataset = apply_train_val_sample_limits(
        dataset, config.get("max_train_samples"), config.get("max_val_samples")
    )
    theta_scaler, x_scaler = checkpoint["theta_scaler"], checkpoint["x_scaler"]
    theta_train = torch.tensor(
        theta_scaler.transform(dataset["train"]["theta"]), dtype=torch.float32
    )
    theta_val = torch.tensor(
        theta_scaler.transform(dataset["val"]["theta"]), dtype=torch.float32
    )
    transform = checkpoint["x_scaling_metadata"]["transform"]
    x_train = prepare_zero_imputed_nre_input(
        x_obs=dataset["train"]["x_obs"], mask=dataset["train"]["mask"],
        x_scaler=x_scaler, transform=transform,
    )
    x_val = prepare_zero_imputed_nre_input(
        x_obs=dataset["val"]["x_obs"], mask=dataset["val"]["mask"],
        x_scaler=x_scaler, transform=transform,
    )
    inference = FixedSplitNRE_C(
        prior=make_scaled_theta_prior("oup", 2, theta_train),
        classifier=config["nre"]["classifier"], device="cpu", show_progress_bars=False,
    )
    inference.append_simulations(torch.cat((theta_train, theta_val)), torch.cat((x_train, x_val)))
    inference.set_fixed_train_val_split(len(theta_train), len(theta_val))
    inference._initialize_neural_network(retrain_from_scratch=False, start_idx=0)
    inference._neural_net.load_state_dict(checkpoint["density_estimator_state_dict"])
    inference._neural_net.eval()
    return checkpoint, inference._neural_net


def load_nre_c_mask_augmentation_checkpoint(checkpoint_path, *, repository_root="."):
    """Rebuild an experimental NRE-C estimator trained on [imputed x, mask]."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    from pathlib import Path
    from gapsbi.datasets import apply_train_val_sample_limits
    from gapsbi.io import load_gapsbi_hdf5
    from gapsbi.preprocessing.scalers import make_scaled_theta_prior

    root, config = Path(repository_root), checkpoint["config"]
    dataset_path = Path(checkpoint["dataset_path"])
    dataset, _ = load_gapsbi_hdf5(
        dataset_path if dataset_path.is_absolute() else root / dataset_path
    )
    dataset = apply_train_val_sample_limits(
        dataset, config.get("max_train_samples"), config.get("max_val_samples")
    )
    theta_scaler, x_scaler = checkpoint["theta_scaler"], checkpoint["x_scaler"]
    theta_train = torch.tensor(
        theta_scaler.transform(dataset["train"]["theta"]), dtype=torch.float32
    )
    theta_val = torch.tensor(
        theta_scaler.transform(dataset["val"]["theta"]), dtype=torch.float32
    )
    transform = checkpoint["x_scaling_metadata"]["transform"]
    x_train = prepare_mask_augmented_nre_input(
        x_obs=dataset["train"]["x_obs"], mask=dataset["train"]["mask"],
        x_scaler=x_scaler, transform=transform,
    )
    x_val = prepare_mask_augmented_nre_input(
        x_obs=dataset["val"]["x_obs"], mask=dataset["val"]["mask"],
        x_scaler=x_scaler, transform=transform,
    )
    inference = FixedSplitNRE_C(
        prior=make_scaled_theta_prior("oup", 2, theta_train),
        classifier=config["nre"]["classifier"], device="cpu", show_progress_bars=False,
    )
    inference.append_simulations(torch.cat((theta_train, theta_val)), torch.cat((x_train, x_val)))
    inference.set_fixed_train_val_split(len(theta_train), len(theta_val))
    inference._initialize_neural_network(retrain_from_scratch=False, start_idx=0)
    inference._neural_net.load_state_dict(checkpoint["density_estimator_state_dict"])
    inference._neural_net.eval()
    return checkpoint, inference._neural_net


@dataclass
class LearnedConstantNRECResult:
    """Joint learned-constant/NRE-C training state."""

    inference: FixedSplitNRE_C
    imputer: torch.nn.Module
    estimator: torch.nn.Module
    training_loss: list[float]
    validation_loss: list[float]
    best_epoch: int
    constant_gradient_observed: bool


def train_learned_constant_imputation_nre_c(
    *, theta_train, x_obs_train_scaled, mask_train, theta_val, x_obs_val_scaled,
    mask_val, prior, classifier, device="cpu", training_batch_size=100,
    num_classes=5, gamma=1.0, learning_rate=5e-4, constant_learning_rate=1e-3,
    stop_after_epochs=20, max_num_epochs=5000, init_value=0.0,
):
    """Jointly optimize canonical learned constants with SBI's NRE-C loss."""
    from gapsbi.methods.learned_constant_imputation import LearnedConstantImputer

    device = torch.device(device)
    if training_batch_size > len(theta_train) or training_batch_size > len(theta_val):
        raise ValueError("training_batch_size exceeds a persisted split.")
    imputer = LearnedConstantImputer(x_obs_train_scaled.shape[1], init_value=init_value).to(device)
    # SBI initializes its ResNet (and internal z-scoring) from the same
    # zero-valued learned-constant initialization used by the NPE method.
    with torch.no_grad():
        x_train_initial = imputer(x_obs_train_scaled.to(device), mask_train.to(device)).cpu()
        x_val_initial = imputer(x_obs_val_scaled.to(device), mask_val.to(device)).cpu()
    inference = FixedSplitNRE_C(prior=prior, classifier=classifier, device=str(device), show_progress_bars=False)
    inference.append_simulations(torch.cat((theta_train, theta_val)), torch.cat((x_train_initial, x_val_initial)))
    inference.set_fixed_train_val_split(len(theta_train), len(theta_val))
    inference._initialize_neural_network(retrain_from_scratch=False, start_idx=0)
    estimator = inference._neural_net.to(device)
    optimizer = torch.optim.Adam([
        {"params": estimator.parameters(), "lr": float(learning_rate)},
        {"params": imputer.parameters(), "lr": float(constant_learning_rate)},
    ])
    train_loader = data.DataLoader(data.TensorDataset(theta_train, x_obs_train_scaled, mask_train), batch_size=int(training_batch_size), shuffle=True, drop_last=True)
    val_loader = data.DataLoader(data.TensorDataset(theta_val, x_obs_val_scaled, mask_val), batch_size=int(training_batch_size), shuffle=False, drop_last=True)
    best_val, best_epoch, stalled = float("inf"), 0, 0
    best_imputer, best_estimator = copy.deepcopy(imputer.state_dict()), copy.deepcopy(estimator.state_dict())
    training_loss, validation_loss, constant_gradient_observed = [], [], False
    for epoch in range(1, int(max_num_epochs) + 1):
        imputer.train(); estimator.train(); train_losses = []
        for theta, x_obs, mask in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = inference._loss(theta.to(device), imputer(x_obs.to(device), mask.to(device)), int(num_classes) + 1, float(gamma))
            loss.backward()
            constant_gradient_observed |= bool(
                imputer.constants.grad is not None
                and torch.isfinite(imputer.constants.grad).all()
                and torch.any(imputer.constants.grad != 0)
            )
            torch.nn.utils.clip_grad_norm_(list(estimator.parameters()) + list(imputer.parameters()), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        imputer.eval(); estimator.eval(); val_losses = []
        with torch.no_grad():
            for theta, x_obs, mask in val_loader:
                loss = inference._loss(theta.to(device), imputer(x_obs.to(device), mask.to(device)), int(num_classes) + 1, float(gamma))
                val_losses.append(float(loss.detach().cpu()))
        train_mean, val_mean = float(np.mean(train_losses)), float(np.mean(val_losses))
        if not np.isfinite(train_mean) or not np.isfinite(val_mean):
            raise FloatingPointError("NRE-C learned-constant loss is nonfinite.")
        training_loss.append(train_mean); validation_loss.append(val_mean)
        if val_mean < best_val:
            best_val, best_epoch, stalled = val_mean, epoch, 0
            best_imputer, best_estimator = copy.deepcopy(imputer.state_dict()), copy.deepcopy(estimator.state_dict())
        else:
            stalled += 1
            if stalled >= int(stop_after_epochs):
                break
    imputer.load_state_dict(best_imputer); estimator.load_state_dict(best_estimator)
    imputer.eval(); estimator.eval()
    return LearnedConstantNRECResult(inference, imputer, estimator, training_loss, validation_loss, best_epoch, constant_gradient_observed)


def load_nre_c_learned_constant_imputation_checkpoint(checkpoint_path, *, repository_root="."):
    """Restore experimental learned constants and their OUP NRE-C classifier."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    from pathlib import Path
    from gapsbi.datasets import apply_train_val_sample_limits
    from gapsbi.io import load_gapsbi_hdf5
    from gapsbi.methods.learned_constant_imputation import LearnedConstantImputer
    from gapsbi.preprocessing.scalers import make_scaled_theta_prior, transform_x_obs_with_fitted_scaler

    root, config = Path(repository_root), checkpoint["config"]
    dataset_path = Path(checkpoint["dataset_path"])
    dataset, _ = load_gapsbi_hdf5(dataset_path if dataset_path.is_absolute() else root / dataset_path)
    dataset = apply_train_val_sample_limits(dataset, config.get("max_train_samples"), config.get("max_val_samples"))
    theta_scaler, x_scaler = checkpoint["theta_scaler"], checkpoint["x_scaler"]
    theta_train = torch.tensor(theta_scaler.transform(dataset["train"]["theta"]), dtype=torch.float32)
    theta_val = torch.tensor(theta_scaler.transform(dataset["val"]["theta"]), dtype=torch.float32)
    transform = checkpoint["x_scaling_metadata"]["transform"]
    def scaled(split):
        return transform_x_obs_with_fitted_scaler(dataset[split]["x_obs"], dataset[split]["mask"], x_scaler, transform)
    imputer = LearnedConstantImputer(int(checkpoint["x_dim"]), float(config["learned_constant_imputation"]["init_value"]))
    imputer.load_state_dict(checkpoint["imputer_state_dict"]); imputer.eval()
    with torch.no_grad():
        x_train = imputer(scaled("train"), torch.as_tensor(dataset["train"]["mask"], dtype=torch.float32))
        x_val = imputer(scaled("val"), torch.as_tensor(dataset["val"]["mask"], dtype=torch.float32))
    inference = FixedSplitNRE_C(prior=make_scaled_theta_prior("oup", 2, theta_train), classifier=config["nre"]["classifier"], device="cpu", show_progress_bars=False)
    inference.append_simulations(torch.cat((theta_train, theta_val)), torch.cat((x_train, x_val)))
    inference.set_fixed_train_val_split(len(theta_train), len(theta_val))
    inference._initialize_neural_network(retrain_from_scratch=False, start_idx=0)
    inference._neural_net.load_state_dict(checkpoint["density_estimator_state_dict"]); inference._neural_net.eval()
    return checkpoint, inference._neural_net, imputer


def evaluate_nre_a_oup_grid(estimator, *, theta_scaler, x_scaled, resolution, batch_size=50_000):
    """Evaluate an NRE-A OUP posterior on its raw uniform-prior midpoint grid."""
    if resolution <= 1 or batch_size <= 0:
        raise ValueError("resolution and batch_size must be positive.")
    t1 = (np.arange(resolution) + .5) * (2.0 / resolution)
    t2 = -2.0 + (np.arange(resolution) + .5) * (5.0 / resolution)
    g1, g2 = np.meshgrid(t1, t2, indexing="ij")
    theta_raw = np.column_stack((g1.ravel(), g2.ravel()))
    theta_scaled = torch.tensor(theta_scaler.transform(theta_raw), dtype=torch.float32)
    x = torch.as_tensor(x_scaled, dtype=torch.float32).reshape(1, -1)
    values = []
    with torch.no_grad():
        for start in range(0, len(theta_scaled), batch_size):
            theta_batch = theta_scaled[start:start + batch_size]
            values.append(estimator.unnormalized_log_ratio(theta_batch, x.expand(len(theta_batch), -1)).cpu().numpy())
    log_ratio = np.concatenate(values).reshape(-1)
    if not np.isfinite(log_ratio).all():
        raise FloatingPointError("NRE-A grid contains nonfinite log-ratios.")
    # log prior is constant (-log(10)) inside this grid. Retain it explicitly:
    # NRE-A estimates log p(x|theta)-log p(x), hence posterior is prior * ratio.
    log_posterior = log_ratio - np.log(10.0)
    log_weights = log_posterior - np.logaddexp.reduce(log_posterior)
    weights = np.exp(log_weights)
    mean = weights @ theta_raw
    centered = theta_raw - mean
    cov = centered.T @ (centered * weights[:, None])
    edge = np.zeros((resolution, resolution), dtype=bool)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    return {"resolution": int(resolution), "theta_raw": theta_raw, "weights": weights,
            "mean": mean, "cov": cov, "boundary_mass": float(weights.reshape(resolution, resolution)[edge].sum()),
            "cell_width": np.array([2.0 / resolution, 5.0 / resolution])}


def sample_nre_a_oup_grid(grid, *, num_samples, seed):
    """Sample raw OUP parameters with uniform within-cell jitter."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(grid["weights"]), size=num_samples, p=grid["weights"])
    draws = grid["theta_raw"][idx] + rng.uniform(-.5, .5, (num_samples, 2)) * grid["cell_width"]
    return draws


class FixedSplitNRE_A(NRE_A):
    """NRE-A whose validation examples are the persisted GapSBI val split."""

    def set_fixed_train_val_split(self, n_train: int, n_val: int) -> "FixedSplitNRE_A":
        if n_train <= 0 or n_val <= 0:
            raise ValueError("n_train and n_val must be positive.")
        # SBI's NRE network initializer also reads these public attributes to
        # fit its internal z-scoring transforms on training examples only.
        self.train_indices = torch.arange(n_train)
        self.val_indices = torch.arange(n_train, n_train + n_val)
        return self

    def get_dataloaders(self, starting_round=0, training_batch_size=200,
                        validation_fraction=0.1, resume_training=False,
                        dataloader_kwargs=None):
        del validation_fraction, resume_training
        if not hasattr(self, "train_indices"):
            raise RuntimeError("Call set_fixed_train_val_split before train().")
        theta, x, prior_masks = self.get_simulations(starting_round)
        dataset = data.TensorDataset(theta, x, prior_masks)
        train_indices, val_indices = self.train_indices, self.val_indices
        if int(val_indices[-1]) >= len(dataset):
            raise ValueError("Fixed split indices exceed appended simulations.")
        if training_batch_size > len(train_indices) or training_batch_size > len(val_indices):
            raise ValueError("training_batch_size exceeds a fixed split.")
        common = dict(dataloader_kwargs or {})
        return (
            data.DataLoader(dataset, batch_size=training_batch_size, drop_last=True,
                            sampler=SubsetRandomSampler(train_indices.tolist()), **common),
            data.DataLoader(dataset, batch_size=training_batch_size, drop_last=True,
                            sampler=SubsetRandomSampler(val_indices.tolist()), **common),
        )


class FixedSplitNRE_C(FixedSplitNRE_A, NRE_C):
    """NRE-C with the same persisted full-data train/validation split."""

    # FixedSplitNRE_A contributes only the dataloader implementation. Explicit
    # dispatch prevents Python's MRO from selecting NRE-A's binary objective.
    train = NRE_C.train
    _loss = NRE_C._loss
    _get_losses = NRE_C._get_losses
