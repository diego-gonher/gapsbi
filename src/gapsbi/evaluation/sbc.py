from __future__ import annotations

import torch


def compute_sbc_ranks_from_samples(
    posterior_samples: torch.Tensor,
    theta_eval: torch.Tensor,
) -> torch.Tensor:
    """Compute SBC ranks from sampled posterior draws.

    posterior_samples: (num_eval, num_posterior_samples, theta_dim)
    theta_eval:         (num_eval, theta_dim)
    returns:            (num_eval, theta_dim)
    """
    return (posterior_samples < theta_eval[:, None, :].cpu()).sum(dim=1)
