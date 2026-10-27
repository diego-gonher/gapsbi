from __future__ import annotations

import torch

from gapsbi.methods.sbi_nre import FixedSplitNRE_A, FixedSplitNRE_C


def test_fixed_split_nre_a_uses_explicit_nonoverlapping_splits():
    prior = torch.distributions.Independent(torch.distributions.Normal(torch.zeros(2), torch.ones(2)), 1)
    inference = FixedSplitNRE_A(prior=prior, classifier="linear", show_progress_bars=False)
    inference.append_simulations(torch.randn(12, 2), torch.randn(12, 3))
    inference.set_fixed_train_val_split(8, 4)
    train, val = inference.get_dataloaders(training_batch_size=4)
    assert set(train.sampler.indices).isdisjoint(val.sampler.indices)
    assert set(train.sampler.indices) == set(range(8))
    assert set(val.sampler.indices) == set(range(8, 12))


def test_fixed_split_nre_c_uses_the_same_persisted_split():
    prior = torch.distributions.Independent(torch.distributions.Normal(torch.zeros(2), torch.ones(2)), 1)
    inference = FixedSplitNRE_C(prior=prior, classifier="linear", show_progress_bars=False)
    inference.append_simulations(torch.randn(12, 2), torch.randn(12, 3))
    inference.set_fixed_train_val_split(8, 4)
    train, val = inference.get_dataloaders(training_batch_size=4)
    assert set(train.sampler.indices).isdisjoint(val.sampler.indices)
