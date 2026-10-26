"""Read-only posterior adapters for saved validated NPE checkpoints.

These adapters intentionally support only the two simple imputation methods
needed by the initial L-C2ST validation. They reconstruct an already-trained
posterior; they never append simulations, train, or write an artifact.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sbi.neural_nets import posterior_nn

from gapsbi.datasets import apply_train_val_sample_limits
from gapsbi.evaluation.posterior_sampling import sample_posteriors_once
from gapsbi.io import load_gapsbi_hdf5
from gapsbi.methods.imputation import zero_impute
from gapsbi.methods.mask_augmentation import make_zero_imputed_mask_augmented_input
from gapsbi.methods.sbi_npe import FixedSplitNPE_C
from gapsbi.preprocessing.scalers import make_scaled_theta_prior, transform_x_obs_with_fitted_scaler


SUPPORTED_METHODS = {"npe_zero_imputation", "npe_mask_augmentation"}
SUPPORTED_PROBLEMS = {"oup", "glm", "glu", "lotka_volterra"}


def _repository_path(path: str | Path, repository_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repository_root / candidate


def _validate_raw_observations(x_obs: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_obs, dtype=np.float64)
    m = np.asarray(mask)
    if x.ndim == 1:
        x = x[None, :]
    if m.ndim == 1:
        m = m[None, :]
    if x.ndim != 2 or x.shape != m.shape:
        raise ValueError("x_obs and mask must have matching shape (observations, features).")
    if not np.all(np.isfinite(x)):
        raise ValueError("x_obs must be finite.")
    if not np.all((m == 0) | (m == 1)):
        raise ValueError("mask must be binary.")
    if not np.array_equal(x[m == 0], np.zeros(np.count_nonzero(m == 0), dtype=x.dtype)):
        raise ValueError("x_obs must be zero at missing entries.")
    return x, m.astype(np.float32, copy=False)


def _scaled_prior(checkpoint: dict[str, Any], repository_root: Path):
    """Rebuild the exact scaled prior, including GLM/LV training covariance."""
    dataset_path = _repository_path(checkpoint["dataset_path"], repository_root)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Checkpoint training dataset is unavailable: {dataset_path}")
    dataset, _metadata = load_gapsbi_hdf5(dataset_path)
    config = checkpoint.get("config", {})
    dataset = apply_train_val_sample_limits(
        dataset,
        max_train_samples=config.get("max_train_samples"),
        max_val_samples=config.get("max_val_samples"),
    )
    theta_train = checkpoint["theta_scaler"].transform(dataset["train"]["theta"])
    theta_train_tensor = torch.as_tensor(theta_train, dtype=torch.float32)
    return make_scaled_theta_prior(
        str(checkpoint["problem"]), int(checkpoint["theta_dim"]), theta_train_tensor
    )


@dataclass
class NPECheckpointAdapter:
    """Reconstructed NPE accepting raw ``(x_obs, mask)`` batches."""

    checkpoint_path: Path
    checkpoint: dict[str, Any]
    posterior: Any
    device: str

    @property
    def method(self) -> str:
        return str(self.checkpoint["method"])

    @property
    def problem(self) -> str:
        return str(self.checkpoint["problem"])

    def prepare_input(self, x_obs: np.ndarray, mask: np.ndarray) -> torch.Tensor:
        """Apply the exact saved scaler and method-specific conditioning input."""
        x, m = _validate_raw_observations(x_obs, mask)
        expected_raw_dim = int(self.checkpoint["x_dim"])
        if self.method == "npe_mask_augmentation":
            expected_raw_dim //= 2
        if x.shape[1] != expected_raw_dim:
            raise ValueError(
                f"Raw feature dimension {x.shape[1]} does not match checkpoint expectation {expected_raw_dim}."
            )
        x_scaled = transform_x_obs_with_fitted_scaler(
            x_obs=x,
            mask=m,
            x_scaler=self.checkpoint["x_scaler"],
            transform=str(self.checkpoint["x_scaling_metadata"]["transform"]),
        )
        mask_tensor = torch.as_tensor(m, dtype=torch.float32)
        if self.method == "npe_zero_imputation":
            conditioned = zero_impute(x_scaled, mask_tensor)
        else:
            conditioned = make_zero_imputed_mask_augmented_input(x_scaled, mask_tensor)
        if conditioned.shape[1] != int(self.checkpoint["x_dim"]):
            raise ValueError("Prepared conditioning dimension does not match checkpoint.")
        return conditioned.to(self.device)

    def sample(
        self,
        x_obs: np.ndarray,
        mask: np.ndarray,
        *,
        num_posterior_samples: int,
        seed: int,
        max_sampling_time: float | None = None,
    ) -> np.ndarray:
        """Return unscaled posterior draws with shape ``(N, draws, theta_dim)``."""
        if num_posterior_samples <= 0:
            raise ValueError("num_posterior_samples must be positive.")
        sampling = self.checkpoint.get("config", {}).get("sampling", {})
        if max_sampling_time is None:
            max_sampling_time = sampling.get("max_sampling_time", 30.0)
        if max_sampling_time is not None:
            max_sampling_time = float(max_sampling_time)
        samples_scaled = sample_posteriors_once(
            posterior=self.posterior,
            x_eval=self.prepare_input(x_obs, mask),
            num_posterior_samples=num_posterior_samples,
            seed=int(seed),
            reject_outside_prior=bool(sampling.get("reject_outside_prior", True)),
            max_sampling_time=max_sampling_time,
        )
        scaled = samples_scaled.detach().cpu().numpy()
        theta_dim = scaled.shape[-1]
        return self.checkpoint["theta_scaler"].inverse_transform(
            scaled.reshape(-1, theta_dim)
        ).reshape(scaled.shape)


def load_npe_checkpoint_adapter(
    checkpoint_path: str | Path,
    *,
    repository_root: str | Path = ".",
    device: str | None = None,
) -> NPECheckpointAdapter:
    """Load a supported checkpoint without altering its training/evaluation artifacts."""
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    method = str(checkpoint.get("method"))
    problem = str(checkpoint.get("problem"))
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported checkpoint method: {method!r}.")
    if problem not in SUPPORTED_PROBLEMS:
        raise ValueError(f"Unsupported checkpoint problem: {problem!r}.")
    for key in ("theta_scaler", "x_scaler", "density_estimator_name", "density_estimator_state_dict"):
        if key not in checkpoint:
            raise ValueError(f"{path}: missing checkpoint field {key!r}.")

    runtime_device = device or str(checkpoint.get("config", {}).get("npe", {}).get("device", "cpu"))
    if runtime_device != "cpu":
        raise ValueError("Checkpoint adapters currently require CPU execution.")
    prior = _scaled_prior(checkpoint, Path(repository_root))
    theta_dim, x_dim = int(checkpoint["theta_dim"]), int(checkpoint["x_dim"])
    density_name = str(checkpoint["density_estimator_name"])
    builder = posterior_nn(density_name)
    density_estimator = builder(
        torch.zeros((2, theta_dim), dtype=torch.float32),
        torch.zeros((2, x_dim), dtype=torch.float32),
    )
    density_estimator.load_state_dict(checkpoint["density_estimator_state_dict"])
    density_estimator.eval()
    inference = FixedSplitNPE_C(prior=prior, density_estimator=density_name, device=runtime_device)
    posterior = inference.build_posterior(density_estimator)
    return NPECheckpointAdapter(path, checkpoint, posterior, runtime_device)
