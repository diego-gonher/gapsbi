from __future__ import annotations

import torch


def compute_tarp_from_samples(
    posterior_samples: torch.Tensor,
    theta_eval: torch.Tensor,
    references: torch.Tensor,
    num_alpha_grid: int = 101,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute TARP ECP curve from sampled posterior draws."""
    theta_eval = theta_eval.detach().cpu()
    references = references.detach().cpu()
    posterior_samples = posterior_samples.detach().cpu()

    dist_true = torch.linalg.norm(theta_eval - references, dim=-1)
    dist_samples = torch.linalg.norm(
        posterior_samples - references[:, None, :],
        dim=-1,
    )

    tarp_probs = (dist_samples < dist_true[:, None]).float().mean(dim=1)
    alpha = torch.linspace(0.0, 1.0, num_alpha_grid)
    ecp = torch.tensor(
        [(tarp_probs <= a).float().mean().item() for a in alpha],
        dtype=torch.float32,
    )
    return ecp, alpha, tarp_probs
