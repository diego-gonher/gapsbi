"""Small fixed-split wrapper for SBI's standard NRE-A trainer."""
from __future__ import annotations

import torch
from sbi.inference import NRE_A
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler


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
    dataset = apply_train_val_sample_limits(dataset, config["max_train_samples"], config["max_val_samples"])
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
