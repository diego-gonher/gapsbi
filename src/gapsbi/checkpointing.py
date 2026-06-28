from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


CHECKPOINT_SCHEMA_VERSION = 1
DEFAULT_CHECKPOINT_NAME = "model_checkpoint.pt"


def save_model_checkpoint(
    path: Path,
    *,
    method: str,
    problem: str,
    seed: int,
    config: dict[str, Any],
    dataset_path: Path,
    theta_scaler: Any,
    x_scaler: Any,
    x_scaling_metadata: dict[str, Any],
    theta_dim: int,
    x_dim: int,
    density_estimator: torch.nn.Module,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save a lightweight per-seed checkpoint for later posterior reconstruction."""
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "method": method,
        "problem": problem,
        "seed": int(seed),
        "config": config,
        "dataset_path": str(dataset_path),
        "theta_scaler": theta_scaler,
        "x_scaler": x_scaler,
        "x_scaling_metadata": x_scaling_metadata,
        "theta_dim": int(theta_dim),
        "x_dim": int(x_dim),
        "density_estimator_class": _module_class_name(density_estimator),
        "density_estimator_state_dict": _to_cpu(density_estimator.state_dict()),
    }
    if extra:
        checkpoint.update(_to_cpu(extra))

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path


def _to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _module_class_name(module: torch.nn.Module) -> str:
    cls = type(module)
    return f"{cls.__module__}.{cls.__qualname__}"
