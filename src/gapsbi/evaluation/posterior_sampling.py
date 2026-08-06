from __future__ import annotations

import inspect

import torch
from tqdm.auto import tqdm

from gapsbi.utils.seeding import set_all_seeds


def sample_posteriors_once(
    posterior,
    x_eval: torch.Tensor,
    num_posterior_samples: int,
    seed: int,
    reject_outside_prior: bool = True,
    max_sampling_time: float | None = 30.0,
    return_num_sampling_failures: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, int, int, bool]:
    """Sample posterior once for every fixed evaluation observation."""
    set_all_seeds(seed)
    posterior_samples = []
    num_sampling_failures = 0
    num_sampling_fallbacks = 0

    sampling_kwargs: dict[str, object] = {
        "reject_outside_prior": bool(reject_outside_prior),
    }
    if max_sampling_time is not None:
        sampling_kwargs["max_sampling_time"] = float(max_sampling_time)

    supported_kwargs = _supported_sampling_kwargs(posterior.sample)
    safe_kwargs = {k: v for k, v in sampling_kwargs.items() if k in supported_kwargs}

    for i in tqdm(range(x_eval.shape[0]), desc="Sampling posteriors"):
        try:
            samples_i = posterior.sample(
                (num_posterior_samples,),
                x=x_eval[i],
                show_progress_bars=False,
                **safe_kwargs,
            )
            samples_i = _validate_posterior_samples(samples_i, x_index=i)
        except (AssertionError, RuntimeError, TypeError, ValueError) as primary_error:
            num_sampling_failures += 1
            num_sampling_fallbacks += 1
            print(f"[posterior_sampling] Sampling failed for x index {i}: {primary_error}")
            print(
                "[posterior_sampling] Filling NaNs "
                f"for x index {i} after sampling failure."
            )
            theta_dim = _infer_theta_dim(posterior, posterior_samples, x_eval)
            samples_i = torch.full(
                (num_posterior_samples, theta_dim),
                float("nan"),
                dtype=torch.float32,
            )
        posterior_samples.append(samples_i.detach().cpu())

    stacked = torch.stack(posterior_samples, dim=0)
    if return_num_sampling_failures:
        return stacked, num_sampling_failures, num_sampling_fallbacks, (
            num_sampling_fallbacks > 0
        )
    return stacked


def _supported_sampling_kwargs(sample_fn) -> set[str]:
    """Return sampling kwargs supported by the installed sbi posterior."""
    try:
        signature = inspect.signature(sample_fn)
    except (TypeError, ValueError):
        return set()

    supports_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in signature.parameters.values()
    )
    if supports_var_kwargs:
        return {"reject_outside_prior", "max_sampling_time"}
    return set(signature.parameters)


def _validate_posterior_samples(samples: torch.Tensor, x_index: int) -> torch.Tensor:
    if not torch.is_tensor(samples):
        samples = torch.as_tensor(samples)
    if not torch.isfinite(samples).all():
        num_bad = int((~torch.isfinite(samples)).sum().item())
        raise ValueError(
            "posterior.sample returned "
            f"{num_bad} non-finite values for x index {x_index}"
        )
    return samples


def _infer_theta_dim(
    posterior,
    collected_samples: list[torch.Tensor],
    x_eval: torch.Tensor,
) -> int:
    if collected_samples:
        return int(collected_samples[0].shape[-1])

    prior = getattr(posterior, "prior", None)
    event_shape = getattr(prior, "event_shape", None)
    if event_shape is not None and len(event_shape) > 0:
        return int(event_shape[-1])

    low = getattr(prior, "low", None)
    if torch.is_tensor(low):
        return int(low.shape[-1])

    # Last-resort fallback: keep shape consistent and avoid crashing.
    return int(x_eval.shape[-1])
