from __future__ import annotations

import torch
from tqdm.auto import tqdm

from gapsbi.utils.seeding import set_all_seeds


def sample_posteriors_once(
    posterior,
    x_eval: torch.Tensor,
    num_posterior_samples: int,
    seed: int,
) -> torch.Tensor:
    """Sample posterior once for every fixed evaluation observation."""
    set_all_seeds(seed)
    posterior_samples = []

    for i in tqdm(range(x_eval.shape[0]), desc="Sampling posteriors"):
        samples_i = posterior.sample(
            (num_posterior_samples,),
            x=x_eval[i],
            show_progress_bars=False,
        )
        posterior_samples.append(samples_i.detach().cpu())

    return torch.stack(posterior_samples, dim=0)
