from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sbi.analysis import plot_summary
from sbi.analysis.plot import sbc_rank_plot
from sbi.diagnostics import check_tarp

from gapsbi.checkpointing import DEFAULT_CHECKPOINT_NAME, save_model_checkpoint
from gapsbi.datasets import apply_train_val_sample_limits
from gapsbi.evaluation.posterior_sampling import sample_posteriors_once
from gapsbi.evaluation.reference_metrics import compute_reference_metrics
from gapsbi.evaluation.sbc import compute_sbc_ranks_from_samples
from gapsbi.evaluation.tarp import compute_tarp_from_samples
from gapsbi.io import load_gapsbi_hdf5
from gapsbi.masks import (
    CoordinateMARMask,
    LotkaVolterraLogTotalMNARMask,
    LotkaVolterraTimeBlockMCARMask,
    LotkaVolterraTimeMARMask,
    MeanNormalizedSelfCensoringMNARMask,
    PointMCARMask,
    SelfCensoringMNARMask,
)
from gapsbi.methods.imputation import (
    compute_observed_feature_means,
    mean_impute,
    zero_impute,
)
from gapsbi.methods.sbi_npe import FixedSplitNPE_C
from gapsbi.preprocessing.scalers import (
    infer_theta_transform,
    infer_x_transform,
    make_scaled_theta_prior,
    scale_theta_train_val_test,
    scale_x_obs_train_val_test_from_full_train,
    theta_scaling_metadata,
    transform_x_obs_with_fitted_scaler,
)
from gapsbi.references import load_reference_posteriors_hdf5
from gapsbi.simulators import GLMSimulator, LotkaVolterraSimulator
from gapsbi.utils.seeding import set_all_seeds

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train NPE baselines with naive imputation on x_obs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} must parse to a mapping/dict.")
    return config


def infer_problem_name(config_problem: str | None, dataset_path: Path) -> str:
    if config_problem:
        return str(config_problem).lower()

    path_lower = str(dataset_path).lower()
    for candidate in ("lotka_volterra", "ricker", "glm", "glu", "oup"):
        if candidate in path_lower:
            return candidate

    raise ValueError(
        "Could not infer problem from dataset path. Please set config['problem'] "
        f"explicitly. dataset_path={dataset_path}"
    )


def ensure_eval_size(test_sample: int, n_test: int) -> None:
    if test_sample <= 0:
        raise ValueError(f"evaluation.test_sample must be positive, got {test_sample}.")
    if test_sample > n_test:
        raise ValueError(
            f"evaluation.test_sample={test_sample} exceeds test split size {n_test}."
        )


def _to_python_scalar(value: Any) -> int | float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, float)):
        return value
    return None


def _extract_training_metadata(inference: Any) -> tuple[int | None, float | None]:
    epochs_trained: int | None = None
    best_validation_loss: float | None = None

    for attr in ("epochs_trained", "_epochs_trained", "num_epochs", "_num_epochs"):
        raw = _to_python_scalar(getattr(inference, attr, None))
        if isinstance(raw, (int, float)):
            epochs_trained = int(raw)
            break

    for attr in (
        "best_validation_loss",
        "_best_validation_loss",
        "_best_val_log_prob",
        "_best_validation_log_prob",
    ):
        raw = _to_python_scalar(getattr(inference, attr, None))
        if isinstance(raw, (int, float)):
            best_validation_loss = float(raw)
            break

    summary = getattr(inference, "summary", None)
    if not isinstance(summary, dict):
        summary = getattr(inference, "_summary", None)

    if isinstance(summary, dict):
        training_loss = summary.get("training_loss")
        validation_loss = summary.get("validation_loss")

        if epochs_trained is None and isinstance(training_loss, list) and training_loss:
            epochs_trained = len(training_loss)

        if (
            best_validation_loss is None
            and isinstance(validation_loss, list)
            and len(validation_loss) > 0
        ):
            best_validation_loss = float(min(validation_loss))

    return epochs_trained, best_validation_loss


def _count_trainable_parameters(module: Any) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def resolve_method_name(imputation_method: str) -> str:
    if imputation_method == "zero":
        return "npe_zero_imputation"
    if imputation_method == "mean":
        return "npe_mean_imputation"
    raise ValueError(
        "imputation.method must be one of {'zero', 'mean'}, "
        f"got {imputation_method!r}."
    )


def sample_tarp_references(
    *,
    problem: str,
    num_references: int,
    seed: int,
    scaled_prior: torch.distributions.Distribution,
    theta_scaler: Any,
    config: dict[str, Any],
) -> torch.Tensor:
    """Sample TARP reference points in the same scaled coordinates as theta_eval."""
    if problem == "glm":
        rng = np.random.default_rng(seed)
        theta = GLMSimulator(
            dim=int(config.get("glm_dim", 10)),
            duration=int(config.get("glm_duration", 100)),
            summary=str(config.get("glm_summary", "raw")),
        ).sample_theta(num_references, rng)
        return torch.tensor(theta_scaler.transform(theta), dtype=torch.float32)

    if problem == "lotka_volterra":
        rng = np.random.default_rng(seed)
        theta = LotkaVolterraSimulator().sample_theta(num_references, rng)
        return torch.tensor(theta_scaler.transform(theta), dtype=torch.float32)

    set_all_seeds(seed)
    return scaled_prior.sample((num_references,)).detach().cpu()


def mask_generator_from_metadata(mask_metadata: dict[str, Any]) -> Any:
    """Rebuild the dataset mask generator for deterministic reference masking."""
    name = str(mask_metadata.get("name", ""))
    missing_fraction = float(mask_metadata["missing_fraction"])

    if name == "point_mcar":
        return PointMCARMask(missing_fraction=missing_fraction)
    if name == "coordinate_mar":
        return CoordinateMARMask(
            missing_fraction=missing_fraction,
            mode=str(mask_metadata.get("mode", "increasing")),
            floor=float(mask_metadata.get("floor", 0.05)),
            max_probability=float(mask_metadata.get("max_probability", 0.95)),
            middle_width=float(mask_metadata.get("middle_width", 0.2)),
        )
    if name == "self_censoring_mnar":
        return SelfCensoringMNARMask(
            missing_fraction=missing_fraction,
            eps=float(mask_metadata.get("eps", 1e-12)),
            score_transform=str(mask_metadata.get("score_transform", "identity")),
        )
    if name == "self_censoring_mnar_mean_normalized":
        return MeanNormalizedSelfCensoringMNARMask(
            missing_fraction=missing_fraction,
            eps=float(mask_metadata.get("eps", 1e-12)),
            score_transform=str(mask_metadata.get("score_transform", "identity")),
        )
    if name == "lv_time_block_mcar":
        return LotkaVolterraTimeBlockMCARMask(
            missing_fraction=missing_fraction,
            block_size=int(mask_metadata.get("block_size", 5)),
            num_populations=int(mask_metadata.get("num_populations", 2)),
        )
    if name == "lv_time_mar":
        return LotkaVolterraTimeMARMask(
            missing_fraction=missing_fraction,
            mode=str(mask_metadata.get("mode", "increasing")),
            floor=float(mask_metadata.get("floor", 0.05)),
            max_probability=float(mask_metadata.get("max_probability", 0.95)),
            middle_width=float(mask_metadata.get("middle_width", 0.2)),
            num_populations=int(mask_metadata.get("num_populations", 2)),
        )
    if name == "lv_log_total_mnar":
        return LotkaVolterraLogTotalMNARMask(
            missing_fraction=missing_fraction,
            eps=float(mask_metadata.get("eps", 1e-12)),
            num_populations=int(mask_metadata.get("num_populations", 2)),
        )

    raise ValueError(f"Unsupported mask generator in dataset metadata: {name!r}.")


def apply_imputation(
    *,
    x_obs_scaled: torch.Tensor,
    mask: torch.Tensor,
    imputation_method: str,
    feature_means: torch.Tensor | None,
) -> torch.Tensor:
    if imputation_method == "zero":
        return zero_impute(x_obs_scaled=x_obs_scaled, mask=mask)
    if feature_means is None:
        raise ValueError("Mean imputation requires feature_means.")
    return mean_impute(
        x_obs_scaled=x_obs_scaled,
        mask=mask,
        feature_means=feature_means,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    dataset_path = Path(config["dataset_path"])
    output_dir = Path(config["output_dir"])
    seeds = [int(s) for s in config["seeds"]]
    max_train_samples = config.get("max_train_samples")
    max_val_samples = config.get("max_val_samples")
    npe_cfg = config["npe"]
    eval_cfg = config["evaluation"]
    sampling_cfg = config.get("sampling", {})
    reference_path = config.get("reference_path")
    reference_mask_seed = int(config.get("reference_mask_seed", 987_654))
    reference_num_posterior_samples = int(
        config.get(
            "reference_num_posterior_samples",
            eval_cfg.get("reference_num_posterior_samples", eval_cfg["num_posterior_samples"]),
        )
    )
    reference_metrics_cfg = config.get("reference_metrics", {})
    c2st_max_samples_per_observation = reference_metrics_cfg.get(
        "c2st_max_samples_per_observation", 2_000
    )
    if c2st_max_samples_per_observation is not None:
        c2st_max_samples_per_observation = int(c2st_max_samples_per_observation)
    c2st_n_folds = int(reference_metrics_cfg.get("c2st_n_folds", 3))
    c2st_hidden_layer_scale = int(reference_metrics_cfg.get("c2st_hidden_layer_scale", 5))
    c2st_max_iter = int(reference_metrics_cfg.get("c2st_max_iter", 10_000))
    reject_outside_prior = bool(sampling_cfg.get("reject_outside_prior", True))
    max_sampling_time = sampling_cfg.get("max_sampling_time", 30.0)
    if max_sampling_time is not None:
        max_sampling_time = float(max_sampling_time)
    imputation_method = str(config["imputation"]["method"]).lower()
    method_name = resolve_method_name(imputation_method)

    output_dir.mkdir(parents=True, exist_ok=True)

    problem = infer_problem_name(config.get("problem"), dataset_path)
    dataset, metadata = load_gapsbi_hdf5(dataset_path)
    dataset = apply_train_val_sample_limits(
        dataset,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
    )

    theta_train_np = dataset["train"]["theta"]
    theta_val_np = dataset["val"]["theta"]
    theta_test_np = dataset["test"]["theta"]

    x_full_train_np = dataset["train"]["x_full"]
    x_obs_train_np = dataset["train"]["x_obs"]
    x_obs_val_np = dataset["val"]["x_obs"]
    x_obs_test_np = dataset["test"]["x_obs"]

    mask_train_np = dataset["train"]["mask"]
    mask_val_np = dataset["val"]["mask"]
    mask_test_np = dataset["test"]["mask"]

    theta_transform = infer_theta_transform(problem)
    theta_train, theta_val, theta_test, theta_scaler = scale_theta_train_val_test(
        theta_train=theta_train_np,
        theta_val=theta_val_np,
        theta_test=theta_test_np,
        transform=theta_transform,
    )
    theta_scaling_metadata_dict = theta_scaling_metadata(theta_transform)

    x_transform = infer_x_transform(problem)
    x_obs_train_scaled, x_obs_val_scaled, x_obs_test_scaled, x_scaler, x_scaling_metadata = (
        scale_x_obs_train_val_test_from_full_train(
            x_full_train=x_full_train_np,
            x_obs_train=x_obs_train_np,
            x_obs_val=x_obs_val_np,
            x_obs_test=x_obs_test_np,
            mask_train=mask_train_np,
            mask_val=mask_val_np,
            mask_test=mask_test_np,
            transform=x_transform,
        )
    )

    mask_train = torch.tensor(mask_train_np, dtype=torch.float32)
    mask_val = torch.tensor(mask_val_np, dtype=torch.float32)
    mask_test = torch.tensor(mask_test_np, dtype=torch.float32)

    feature_means = None
    if imputation_method == "zero":
        x_train = zero_impute(x_obs_scaled=x_obs_train_scaled, mask=mask_train)
        x_val = zero_impute(x_obs_scaled=x_obs_val_scaled, mask=mask_val)
        x_test = zero_impute(x_obs_scaled=x_obs_test_scaled, mask=mask_test)
    else:
        feature_means = compute_observed_feature_means(
            x_obs_scaled_train=x_obs_train_scaled,
            mask_train=mask_train,
        )
        x_train = mean_impute(
            x_obs_scaled=x_obs_train_scaled,
            mask=mask_train,
            feature_means=feature_means,
        )
        x_val = mean_impute(
            x_obs_scaled=x_obs_val_scaled,
            mask=mask_val,
            feature_means=feature_means,
        )
        x_test = mean_impute(
            x_obs_scaled=x_obs_test_scaled,
            mask=mask_test,
            feature_means=feature_means,
        )

    theta_dim = theta_train.shape[1]
    x_dim = x_train.shape[1]
    num_train_examples = theta_train.shape[0]
    num_val_examples = theta_val.shape[0]
    num_test_examples = theta_test.shape[0]
    prior = make_scaled_theta_prior(problem, theta_dim, theta_train_scaled=theta_train)

    test_sample = int(eval_cfg["test_sample"])
    num_posterior_samples = int(eval_cfg["num_posterior_samples"])
    num_alpha_grid = int(eval_cfg["num_alpha_grid"])

    ensure_eval_size(test_sample=test_sample, n_test=theta_test.shape[0])

    all_results: list[dict[str, Any]] = []

    for seed in seeds:
        print("\n" + "=" * 80)
        print(f"Running seed {seed}")
        print("=" * 80)

        set_all_seeds(seed)
        seed_output_dir = output_dir / f"seed_{seed}"
        seed_output_dir.mkdir(parents=True, exist_ok=True)

        # Total seed runtime (train + posterior sampling + diagnostics + saving).
        total_start = time.perf_counter()

        theta_trainval = torch.cat([theta_train, theta_val], dim=0)
        x_trainval = torch.cat([x_train, x_val], dim=0)
        inference = FixedSplitNPE_C(
            prior=prior,
            density_estimator=npe_cfg["density_estimator"],
            device=str(npe_cfg.get("device", "cpu")),
        )
        inference.append_simulations(theta_trainval, x_trainval)
        inference.set_fixed_train_val_split(
            n_train=num_train_examples,
            n_val=num_val_examples,
        )

        # Time only the call to inference.train(...).
        training_start = time.perf_counter()
        density_estimator = inference.train(
            training_batch_size=int(npe_cfg["training_batch_size"]),
            validation_fraction=0.1,  # ignored by FixedSplitNPE_C
            stop_after_epochs=int(npe_cfg["stop_after_epochs"]),
            max_num_epochs=int(npe_cfg["max_num_epochs"]),
        )
        training_end = time.perf_counter()
        training_time_sec = training_end - training_start

        posterior = inference.build_posterior(density_estimator)
        epochs_trained, best_validation_loss = _extract_training_metadata(inference)
        num_parameters = _count_trainable_parameters(density_estimator)
        model_checkpoint_path = save_model_checkpoint(
            seed_output_dir / DEFAULT_CHECKPOINT_NAME,
            method=method_name,
            problem=problem,
            seed=seed,
            config=config,
            dataset_path=dataset_path,
            theta_scaler=theta_scaler,
            theta_scaling_metadata=theta_scaling_metadata_dict,
            x_scaler=x_scaler,
            x_scaling_metadata=x_scaling_metadata,
            theta_dim=theta_dim,
            x_dim=x_dim,
            density_estimator=density_estimator,
            extra={
                "config_path": str(args.config),
                "density_estimator_name": str(npe_cfg["density_estimator"]),
                "epochs_trained": epochs_trained,
                "best_validation_loss": best_validation_loss,
                "imputation_method": imputation_method,
            },
        )

        _ = plot_summary(
            inference,
            tags=["training_loss", "validation_loss"],
            figsize=(10, 2),
        )
        training_summary_path = seed_output_dir / "training_summary.png"
        plt.savefig(training_summary_path, dpi=200, bbox_inches="tight")
        plt.close()

        theta_eval = theta_test[:test_sample]
        x_eval = x_test[:test_sample]

        # Time posterior sampling only.
        sampling_start = time.perf_counter()
        (
            posterior_samples_scaled,
            num_sampling_failures,
            num_sampling_fallbacks,
            fallback_sampling_used,
        ) = sample_posteriors_once(
            posterior=posterior,
            x_eval=x_eval,
            num_posterior_samples=num_posterior_samples,
            seed=seed + 10_000,
            reject_outside_prior=reject_outside_prior,
            max_sampling_time=max_sampling_time,
            return_num_sampling_failures=True,
        )
        sampling_end = time.perf_counter()
        posterior_sampling_time_sec = sampling_end - sampling_start

        posterior_samples_scaled_np = posterior_samples_scaled.detach().cpu().numpy()
        theta_eval_scaled_np = theta_eval.detach().cpu().numpy()
        num_eval, num_samples, theta_dim_local = posterior_samples_scaled_np.shape

        posterior_samples_np = theta_scaler.inverse_transform(
            posterior_samples_scaled_np.reshape(-1, theta_dim_local)
        ).reshape(num_eval, num_samples, theta_dim_local)
        theta_eval_np = theta_scaler.inverse_transform(theta_eval_scaled_np)

        posterior_h5_path = seed_output_dir / "posterior_samples.h5"
        with h5py.File(posterior_h5_path, "w") as f:
            f.create_dataset("theta_true", data=theta_eval_np)
            f.create_dataset("theta_true_scaled", data=theta_eval_scaled_np)
            f.create_dataset("theta_posterior", data=posterior_samples_np)
            f.create_dataset("theta_posterior_scaled", data=posterior_samples_scaled_np)

            f.attrs["dataset_path"] = str(dataset_path)
            f.attrs["problem"] = problem
            f.attrs["method"] = method_name
            f.attrs["seed"] = int(seed)
            f.attrs["num_eval"] = int(num_eval)
            f.attrs["num_posterior_samples"] = int(num_samples)
            f.attrs["theta_dim"] = int(theta_dim_local)
            f.attrs["theta_transform"] = theta_scaling_metadata_dict["transform"]
            f.attrs["x_transform"] = x_scaling_metadata["transform"]

        # Time diagnostics: TARP/SBC computations and diagnostic artifacts.
        diagnostics_start = time.perf_counter()

        references = sample_tarp_references(
            problem=problem,
            num_references=test_sample,
            seed=seed + 20_000,
            scaled_prior=prior,
            theta_scaler=theta_scaler,
            config=config,
        )
        ecp, alpha, tarp_probs = compute_tarp_from_samples(
            posterior_samples=posterior_samples_scaled,
            theta_eval=theta_eval,
            references=references,
            num_alpha_grid=num_alpha_grid,
        )
        atc, ks_pval = check_tarp(ecp.detach().cpu(), alpha.detach().cpu())
        print(f"TARP ATC: {atc:.4f}")
        print(f"TARP KS p-value: {ks_pval:.4f}")

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(alpha.numpy(), ecp.numpy(), label="TARP ECP")
        ax.plot([0, 1], [0, 1], linestyle="--", label="ideal")
        ax.set_xlabel("Nominal coverage")
        ax.set_ylabel("Expected coverage probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.legend()
        tarp_plot_path = seed_output_dir / "tarp.png"
        fig.savefig(tarp_plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        ranks = compute_sbc_ranks_from_samples(
            posterior_samples=posterior_samples_scaled,
            theta_eval=theta_eval,
        )
        fig, _ax = sbc_rank_plot(
            ranks,
            num_posterior_samples,
            num_bins=20,
            figsize=(8, 3),
            plot_type="hist",
        )
        sbc_plot_path = seed_output_dir / "sbc_rank_histograms.png"
        fig.savefig(sbc_plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        diagnostics_arrays_path = seed_output_dir / "diagnostics_arrays.npz"
        np.savez(
            diagnostics_arrays_path,
            alpha=alpha.detach().cpu().numpy(),
            ecp=ecp.detach().cpu().numpy(),
            tarp_probs=tarp_probs.detach().cpu().numpy(),
            references_scaled=references.detach().cpu().numpy(),
            ranks=ranks.detach().cpu().numpy(),
        )

        summary = {
            "seed": int(seed),
            "problem": problem,
            "method": method_name,
            "dataset_path": str(dataset_path),
            "theta_transform": theta_scaling_metadata_dict["transform"],
            "x_transform": x_scaling_metadata["transform"],
            "training_time_sec": float(training_time_sec),
            "posterior_sampling_time_sec": float(posterior_sampling_time_sec),
            "epochs_trained": epochs_trained,
            "best_validation_loss": best_validation_loss,
            "num_train_examples": int(num_train_examples),
            "num_val_examples": int(num_val_examples),
            "num_test_examples": int(num_test_examples),
            "max_train_samples": max_train_samples,
            "max_val_samples": max_val_samples,
            "theta_dim": int(theta_dim),
            "x_dim": int(x_dim),
            "device": str(npe_cfg.get("device", "cpu")),
            "density_estimator": str(npe_cfg["density_estimator"]),
            "num_parameters": int(num_parameters),
            "imputation_method": imputation_method,
            "uses_mask": False,
            "x_source": "x_obs",
            "x_scaler_fit_source": "x_full_train",
            "imputation_space": "scaled",
            "mask_convention": "1=observed,0=missing",
            "reject_outside_prior": bool(reject_outside_prior),
            "max_sampling_time": max_sampling_time,
            "num_sampling_failures": int(num_sampling_failures),
            "num_sampling_fallbacks": int(num_sampling_fallbacks),
            "fallback_sampling_used": bool(fallback_sampling_used),
            "tarp_atc": float(atc),
            "tarp_ks_pvalue": float(ks_pval),
            "tarp_reference_space": "theta_scaled",
            "tarp_reference_distribution": (
                "true_simulator_prior_scaled"
                if problem in {"glm", "lotka_volterra"}
                else "scaled_npe_prior"
            ),
            "num_tarp_eval": int(test_sample),
            "num_tarp_posterior_samples": int(num_posterior_samples),
            "num_sbc_eval": int(test_sample),
            "num_sbc_posterior_samples": int(num_posterior_samples),
            "posterior_samples_path": str(posterior_h5_path),
            "model_checkpoint_path": str(model_checkpoint_path),
            "training_summary_path": str(training_summary_path),
            "tarp_plot_path": str(tarp_plot_path),
            "sbc_plot_path": str(sbc_plot_path),
            "diagnostics_arrays_path": str(diagnostics_arrays_path),
        }

        reference_sampling_time_sec: float | None = None
        reference_posterior_samples_path: Path | None = None
        reference_num_sampling_failures: int | None = None
        reference_num_sampling_fallbacks: int | None = None
        reference_fallback_sampling_used: bool | None = None
        if reference_path:
            ref_observations, ref_theta_samples, ref_metadata, _ref_diagnostics = (
                load_reference_posteriors_hdf5(reference_path)
            )
            if str(ref_metadata.get("problem", "")).lower() != problem:
                raise ValueError(
                    f"Reference problem {ref_metadata.get('problem')!r} does not "
                    f"match config problem {problem!r}."
                )
            if "mask" not in metadata or not isinstance(metadata["mask"], dict):
                raise ValueError(
                    "Dataset metadata must include a mask dictionary to evaluate "
                    "reference observations under imputation."
                )

            ref_mask_generator = mask_generator_from_metadata(metadata["mask"])
            ref_rng = np.random.default_rng(reference_mask_seed)
            ref_mask = ref_mask_generator.generate(
                x_full=ref_observations["x_full"],
                theta=ref_observations["theta_true"],
                rng=ref_rng,
            )
            ref_x_obs = ref_observations["x_full"] * ref_mask
            ref_x_obs_scaled = transform_x_obs_with_fitted_scaler(
                x_obs=ref_x_obs,
                mask=ref_mask,
                x_scaler=x_scaler,
                transform=x_transform,
            )
            ref_mask_tensor = torch.tensor(ref_mask, dtype=torch.float32)
            x_ref = apply_imputation(
                x_obs_scaled=ref_x_obs_scaled,
                mask=ref_mask_tensor,
                imputation_method=imputation_method,
                feature_means=feature_means,
            )

            print(
                "Reference masks: "
                f"missing_fraction={1.0 - float(np.mean(ref_mask)):.4f}, "
                f"seed={reference_mask_seed}"
            )

            reference_sampling_start = time.perf_counter()
            (
                reference_samples_scaled,
                reference_num_sampling_failures,
                reference_num_sampling_fallbacks,
                reference_fallback_sampling_used,
            ) = sample_posteriors_once(
                posterior=posterior,
                x_eval=x_ref,
                num_posterior_samples=reference_num_posterior_samples,
                seed=seed + 30_000,
                reject_outside_prior=reject_outside_prior,
                max_sampling_time=max_sampling_time,
                return_num_sampling_failures=True,
            )
            reference_sampling_end = time.perf_counter()
            reference_sampling_time_sec = reference_sampling_end - reference_sampling_start

            reference_samples_scaled_np = reference_samples_scaled.detach().cpu().numpy()
            num_ref_eval, num_ref_samples, ref_theta_dim = reference_samples_scaled_np.shape
            reference_samples_np = theta_scaler.inverse_transform(
                reference_samples_scaled_np.reshape(-1, ref_theta_dim)
            ).reshape(num_ref_eval, num_ref_samples, ref_theta_dim)
            if reference_samples_np.shape != ref_theta_samples.shape:
                raise ValueError(
                    "Estimator reference posterior samples must match reference "
                    f"posterior shape exactly, got estimator={reference_samples_np.shape} "
                    f"and reference={ref_theta_samples.shape}."
                )

            reference_posterior_samples_path = seed_output_dir / "reference_posterior_samples.h5"
            with h5py.File(reference_posterior_samples_path, "w") as f:
                f.create_dataset("theta_true", data=ref_observations["theta_true"])
                f.create_dataset("x_full", data=ref_observations["x_full"])
                f.create_dataset("mask", data=ref_mask)
                f.create_dataset("x_obs", data=ref_x_obs)
                f.create_dataset("theta_posterior", data=reference_samples_np)
                f.create_dataset("theta_posterior_scaled", data=reference_samples_scaled_np)
                f.attrs["dataset_path"] = str(dataset_path)
                f.attrs["reference_path"] = str(reference_path)
                f.attrs["problem"] = problem
                f.attrs["method"] = method_name
                f.attrs["seed"] = int(seed)
                f.attrs["reference_mask_seed"] = int(reference_mask_seed)
                f.attrs["reference_mask_missing_fraction"] = float(1.0 - np.mean(ref_mask))
                f.attrs["reference_metric_target"] = "masked_imputed_x_vs_full_x_reference"
                f.attrs["num_eval"] = int(num_ref_eval)
                f.attrs["num_posterior_samples"] = int(num_ref_samples)
                f.attrs["theta_dim"] = int(ref_theta_dim)
                f.attrs["theta_transform"] = theta_scaling_metadata_dict["transform"]
                f.attrs["x_transform"] = x_scaling_metadata["transform"]

            reference_metrics_start = time.perf_counter()
            reference_metric_rows = compute_reference_metrics(
                estimator_samples=reference_samples_np,
                reference_samples=ref_theta_samples,
                seed=seed + 40_000,
                max_samples_per_observation=num_ref_samples,
                c2st_max_samples_per_observation=c2st_max_samples_per_observation,
                c2st_n_folds=c2st_n_folds,
                c2st_hidden_layer_scale=c2st_hidden_layer_scale,
                c2st_max_iter=c2st_max_iter,
                progress=True,
            )
            reference_metrics_time_sec = time.perf_counter() - reference_metrics_start

            summary.update(
                {
                    "reference_path": str(reference_path),
                    "reference_posterior_samples_path": str(reference_posterior_samples_path),
                    "reference_metric_target": "masked_imputed_x_vs_full_x_reference",
                    "reference_mask_seed": int(reference_mask_seed),
                    "reference_mask_name": str(metadata["mask"].get("name", "")),
                    "reference_mask_missing_fraction": float(1.0 - np.mean(ref_mask)),
                    "num_reference_eval": int(num_ref_eval),
                    "num_reference_posterior_samples": int(num_ref_samples),
                    "reference_posterior_sampling_time_sec": float(reference_sampling_time_sec),
                    "reference_num_sampling_failures": int(reference_num_sampling_failures),
                    "reference_num_sampling_fallbacks": int(reference_num_sampling_fallbacks),
                    "reference_fallback_sampling_used": bool(reference_fallback_sampling_used),
                    "reference_metrics_time_sec": float(reference_metrics_time_sec),
                    "reference_metrics_space": "theta_unscaled",
                    "reference_metrics_num_samples_used": [
                        int(row["num_samples_used"]) for row in reference_metric_rows
                    ],
                    "reference_metrics_c2st_num_samples_used": [
                        int(row["c2st_num_samples_used"]) for row in reference_metric_rows
                    ],
                    "reference_metrics_c2st_n_folds": int(c2st_n_folds),
                    "reference_metrics_c2st_hidden_layer_scale": int(c2st_hidden_layer_scale),
                    "reference_metrics_c2st_max_iter": int(c2st_max_iter),
                    "reference_metrics_reference_index": [
                        int(row["reference_index"]) for row in reference_metric_rows
                    ],
                    "reference_c2st_accuracy": [
                        float(row["c2st_accuracy"]) for row in reference_metric_rows
                    ],
                    "reference_posterior_mean_shift": [
                        float(row["posterior_mean_shift"]) for row in reference_metric_rows
                    ],
                    "reference_covariance_trace_ratio": [
                        float(row["covariance_trace_ratio"]) for row in reference_metric_rows
                    ],
                }
            )
        diagnostics_end = time.perf_counter()
        diagnostics_time_sec = diagnostics_end - diagnostics_start
        total_end = time.perf_counter()
        total_runtime_sec = total_end - total_start

        summary["diagnostics_time_sec"] = float(diagnostics_time_sec)
        summary["total_runtime_sec"] = float(total_runtime_sec)

        with (seed_output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        all_results.append(summary)

    with (output_dir / "all_results.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nFinished all seeds. Results: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
