from __future__ import annotations

from typing import Dict, Literal, Optional, Union

import torch
from sbi.inference import NPE_C
from sbi.neural_nets.estimators.base import (
    ConditionalDensityEstimator,
    ConditionalEstimatorBuilder,
)
from sbi.sbi_types import Tracker
from torch.distributions import Distribution
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard.writer import SummaryWriter


class FixedSplitNPE_C(NPE_C):
    """NPE_C with explicit user-controlled train/validation splits."""

    def __init__(
        self,
        prior: Optional[Distribution] = None,
        density_estimator: Union[
            Literal["nsf", "maf", "mdn", "made"],
            ConditionalEstimatorBuilder[ConditionalDensityEstimator],
        ] = "nsf",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        tracker: Optional[Tracker] = None,
        show_progress_bars: bool = True,
    ) -> None:
        super().__init__(
            prior=prior,
            density_estimator=density_estimator,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            tracker=tracker,
            show_progress_bars=show_progress_bars,
        )
        self._use_fixed_split = False
        self.train_indices: Optional[torch.Tensor] = None
        self.val_indices: Optional[torch.Tensor] = None

    def set_fixed_train_val_split(
        self,
        n_train: int,
        n_val: int,
        device: Optional[torch.device] = None,
    ) -> "FixedSplitNPE_C":
        if n_train <= 0:
            raise ValueError(f"n_train must be positive, got {n_train}.")
        if n_val <= 0:
            raise ValueError(f"n_val must be positive, got {n_val}.")

        if device is None:
            device = torch.device("cpu")

        self.train_indices = torch.arange(0, n_train, dtype=torch.long, device=device)
        self.val_indices = torch.arange(
            n_train, n_train + n_val, dtype=torch.long, device=device
        )
        self._n_train_fixed = n_train
        self._n_val_fixed = n_val
        self._use_fixed_split = True
        return self

    def set_fixed_train_val_indices(
        self,
        train_indices: torch.Tensor,
        val_indices: torch.Tensor,
    ) -> "FixedSplitNPE_C":
        train_indices = torch.as_tensor(train_indices, dtype=torch.long).detach().cpu()
        val_indices = torch.as_tensor(val_indices, dtype=torch.long).detach().cpu()

        if train_indices.numel() == 0:
            raise ValueError("train_indices is empty.")
        if val_indices.numel() == 0:
            raise ValueError("val_indices is empty.")
        if torch.isin(train_indices, val_indices).any():
            raise ValueError("train_indices and val_indices overlap.")

        self.train_indices = train_indices
        self.val_indices = val_indices
        self._n_train_fixed = train_indices.numel()
        self._n_val_fixed = val_indices.numel()
        self._use_fixed_split = True
        return self

    def get_dataloaders(
        self,
        starting_round: int = 0,
        training_batch_size: int = 200,
        validation_fraction: float = 0.1,
        resume_training: bool = False,
        dataloader_kwargs: Optional[Dict] = None,
    ):
        del validation_fraction, resume_training
        if not self._use_fixed_split:
            raise RuntimeError(
                "Fixed splits have not been set. Call either "
                ".set_fixed_train_val_split(n_train, n_val) or "
                ".set_fixed_train_val_indices(train_indices, val_indices) "
                "before calling .train()."
            )
        assert self.train_indices is not None
        assert self.val_indices is not None

        theta, x, prior_masks = self.get_simulations(starting_round)
        dataset = data.TensorDataset(theta, x, prior_masks)

        num_examples = theta.shape[0]
        max_train_idx = int(self.train_indices.max().item())
        max_val_idx = int(self.val_indices.max().item())
        if max(max_train_idx, max_val_idx) >= num_examples:
            raise ValueError(
                "Fixed split indices exceed the number of appended simulations. "
                f"Got max index {max(max_train_idx, max_val_idx)}, but only "
                f"{num_examples} simulations were appended."
            )

        num_training_examples = len(self.train_indices)
        num_validation_examples = len(self.val_indices)
        if num_training_examples < training_batch_size:
            raise ValueError(
                f"training_batch_size={training_batch_size} is larger than the "
                f"number of training examples={num_training_examples}."
            )
        if num_validation_examples < training_batch_size:
            raise ValueError(
                f"training_batch_size={training_batch_size} is larger than the "
                f"number of validation examples={num_validation_examples}."
            )

        train_loader_kwargs = {
            "batch_size": min(training_batch_size, num_training_examples),
            "drop_last": True,
            "sampler": SubsetRandomSampler(self.train_indices.cpu().tolist()),
        }
        val_loader_kwargs = {
            "batch_size": min(training_batch_size, num_validation_examples),
            "shuffle": False,
            "drop_last": True,
            "sampler": SubsetRandomSampler(self.val_indices.cpu().tolist()),
        }
        if dataloader_kwargs is not None:
            train_loader_kwargs = dict(train_loader_kwargs, **dataloader_kwargs)
            val_loader_kwargs = dict(val_loader_kwargs, **dataloader_kwargs)

        train_loader = data.DataLoader(dataset, **train_loader_kwargs)
        val_loader = data.DataLoader(dataset, **val_loader_kwargs)
        return train_loader, val_loader


def train_fixed_split_npe(
    theta_train: torch.Tensor,
    x_train: torch.Tensor,
    theta_val: torch.Tensor,
    x_val: torch.Tensor,
    prior: Distribution,
    density_estimator: Union[
        Literal["nsf", "maf", "mdn", "made"],
        ConditionalEstimatorBuilder[ConditionalDensityEstimator],
    ] = "nsf",
    device: str = "cpu",
    training_batch_size: int = 256,
    stop_after_epochs: int = 20,
    max_num_epochs: int = 500,
) -> tuple[FixedSplitNPE_C, ConditionalDensityEstimator, object]:
    """Train NPE_C with predefined train/validation splits.

    Concatenation order exactly follows the notebook logic:
    `[theta_train, theta_val]` and `[x_train, x_val]`.
    """

    theta_trainval = torch.cat([theta_train, theta_val], dim=0)
    x_trainval = torch.cat([x_train, x_val], dim=0)

    inference = FixedSplitNPE_C(
        prior=prior,
        density_estimator=density_estimator,
        device=device,
    )
    inference.append_simulations(theta_trainval, x_trainval)
    inference.set_fixed_train_val_split(
        n_train=theta_train.shape[0],
        n_val=theta_val.shape[0],
    )

    trained_density_estimator = inference.train(
        training_batch_size=training_batch_size,
        validation_fraction=0.1,  # ignored by FixedSplitNPE_C
        stop_after_epochs=stop_after_epochs,
        max_num_epochs=max_num_epochs,
    )
    posterior = inference.build_posterior(trained_density_estimator)
    return inference, trained_density_estimator, posterior
