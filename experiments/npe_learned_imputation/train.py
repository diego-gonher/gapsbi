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
from sbi.analysis.plot import sbc_rank_plot
from sbi.diagnostics import check_tarp
from sbi.utils import BoxUniform

from gapsbi.datasets import apply_train_val_sample_limits
from gapsbi.evaluation.posterior_sampling import sample_posteriors_once
from gapsbi.evaluation.sbc import compute_sbc_ranks_from_samples
from gapsbi.evaluation.tarp import compute_tarp_from_samples
from gapsbi.io import load_gapsbi_hdf5
from gapsbi.methods.learned_imputation import (
    parse_missingness_and_epsilon,
    prepare_learned_imputation_arrays,
    train_learned_imputation_npe,
)
from gapsbi.methods.sbi_npe import FixedSplitNPE_C
from gapsbi.utils.seeding import set_all_seeds

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train learned deterministic imputation + NPE baseline."
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
    for candidate in ("ricker", "glm", "glu", "oup"):
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


def _count_trainable_parameters(*modules: torch.nn.Module) -> int:
    count = 0
    for module in modules:
        count += int(sum(p.numel() for p in module.parameters() if p.requires_grad))
    return count


def _resolve_imputer_type(problem: str, imputer_cfg: dict[str, Any]) -> str:
    requested = str(imputer_cfg.get("imputer_type", "auto")).lower()
    if requested == "auto":
        return "cnn" if problem in {"oup", "ricker"} else "mlp"
    return requested


def _plot_training_history(
    history: list[dict[str, float]],
    output_path: Path,
) -> None:
    if not history:
        return
    epochs = [int(h["epoch"]) for h in history]
    train_total = [float(h["train_total_loss"]) for h in history]
    val_total = [float(h["val_total_loss"]) for h in history]
    train_npe = [float(h["train_npe_loss"]) for h in history]
    val_npe = [float(h["val_npe_loss"]) for h in history]
    train_recon = [float(h["train_recon_loss"]) for h in history]
    val_recon = [float(h["val_recon_loss"]) for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    axes[0].plot(epochs, train_total, label="train")
    axes[0].plot(epochs, val_total, label="val")
    axes[0].set_title("Total loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, train_npe, label="train")
    axes[1].plot(epochs, val_npe, label="val")
    axes[1].set_title("NPE loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(epochs, train_recon, label="train")
    axes[2].plot(epochs, val_recon, label="val")
    axes[2].set_title("Recon loss")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _complete_with_imputer(
    imputer: torch.nn.Module,
    x_obs_scaled: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    with torch.no_grad():
        x_hat = imputer(x_obs_scaled.to(device), mask.to(device))
        x_completed = mask.to(device) * x_obs_scaled.to(device) + (
            1.0 - mask.to(device)
        ) * x_hat
    return x_completed.detach().cpu()


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
    imputer_cfg = config["imputer"]
    sampling_cfg = config.get("sampling", {})
    reject_outside_prior = bool(sampling_cfg.get("reject_outside_prior", False))
    max_sampling_time = sampling_cfg.get("max_sampling_time", None)
    if max_sampling_time is not None:
        max_sampling_time = float(max_sampling_time)

    output_dir.mkdir(parents=True, exist_ok=True)

    problem = infer_problem_name(config.get("problem"), dataset_path)
    mechanism, epsilon = parse_missingness_and_epsilon(dataset_path)
    dataset, _metadata = load_gapsbi_hdf5(dataset_path)
    dataset = apply_train_val_sample_limits(
        dataset,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
    )
    prepared = prepare_learned_imputation_arrays(dataset=dataset, problem=problem)

    theta_train = prepared["theta_train"]
    theta_val = prepared["theta_val"]
    theta_test = prepared["theta_test"]
    x_full_train_scaled = prepared["x_full_train_scaled"]
    x_full_val_scaled = prepared["x_full_val_scaled"]
    x_obs_train_scaled = prepared["x_obs_train_scaled"]
    x_obs_val_scaled = prepared["x_obs_val_scaled"]
    x_obs_test_scaled = prepared["x_obs_test_scaled"]
    mask_train = prepared["mask_train"]
    mask_val = prepared["mask_val"]
    mask_test = prepared["mask_test"]
    theta_scaler = prepared["theta_scaler"]
    x_scaling_metadata = prepared["x_scaling_metadata"]

    theta_dim = theta_train.shape[1]
    x_dim = x_obs_train_scaled.shape[1]
    prior = BoxUniform(low=-torch.ones(theta_dim), high=torch.ones(theta_dim))

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

        total_start = time.perf_counter()
        status = "ok"
        try:
            training_start = time.perf_counter()
            learned = train_learned_imputation_npe(
                problem=problem,
                theta_train=theta_train,
                x_obs_train_scaled=x_obs_train_scaled,
                x_full_train_scaled=x_full_train_scaled,
                mask_train=mask_train,
                theta_val=theta_val,
                x_obs_val_scaled=x_obs_val_scaled,
                x_full_val_scaled=x_full_val_scaled,
                mask_val=mask_val,
                prior=prior,
                config={**npe_cfg, **imputer_cfg},
            )
            training_end = time.perf_counter()
            training_time_sec = training_end - training_start

            training_summary_path = seed_output_dir / "training_summary.png"
            _plot_training_history(learned.train_history, training_summary_path)

            device = torch.device(str(npe_cfg.get("device", "cpu")))
            x_test_completed = _complete_with_imputer(
                learned.imputer,
                x_obs_test_scaled,
                mask_test,
                device,
            )
            theta_eval = theta_test[:test_sample]
            x_eval = x_test_completed[:test_sample]

            sampling_start = time.perf_counter()
            (
                posterior_samples_scaled,
                num_sampling_failures,
                num_sampling_fallbacks,
                fallback_sampling_used,
            ) = sample_posteriors_once(
                posterior=learned.posterior,
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
                f.attrs["method"] = "npe_learned_imputation"
                f.attrs["seed"] = int(seed)
                f.attrs["num_eval"] = int(num_eval)
                f.attrs["num_posterior_samples"] = int(num_samples)
                f.attrs["theta_dim"] = int(theta_dim_local)
                f.attrs["x_transform"] = x_scaling_metadata["transform"]

            diagnostics_start = time.perf_counter()
            set_all_seeds(seed + 20_000)
            references = prior.sample((test_sample,)).detach().cpu()
            ecp, alpha, tarp_probs = compute_tarp_from_samples(
                posterior_samples=posterior_samples_scaled,
                theta_eval=theta_eval,
                references=references,
                num_alpha_grid=num_alpha_grid,
            )
            atc, ks_pval = check_tarp(ecp.detach().cpu(), alpha.detach().cpu())

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

            diagnostics_end = time.perf_counter()
            diagnostics_time_sec = diagnostics_end - diagnostics_start
            total_end = time.perf_counter()
            total_runtime_sec = total_end - total_start

            summary = {
                "seed": int(seed),
                "problem": problem,
                "missingness": mechanism,
                "epsilon": epsilon,
                "method": "npe_learned_imputation",
                "dataset_path": str(dataset_path),
                "x_transform": x_scaling_metadata["transform"],
                "x_dim": int(x_dim),
                "theta_dim": int(theta_dim),
                "num_train_examples": int(theta_train.shape[0]),
                "num_val_examples": int(theta_val.shape[0]),
                "num_test_examples": int(theta_test.shape[0]),
                "max_train_samples": max_train_samples,
                "max_val_samples": max_val_samples,
                "lambda_recon": float(imputer_cfg.get("lambda_recon", 1.0)),
                "imputer_type": _resolve_imputer_type(problem, imputer_cfg),
                "density_estimator": str(npe_cfg.get("density_estimator", "nsf")),
                "best_val_loss": float(learned.best_val_loss),
                "best_epoch": int(learned.best_epoch),
                "stopped_epoch": int(learned.stopped_epoch),
                "num_parameters": int(
                    _count_trainable_parameters(
                        learned.imputer,
                        learned.density_estimator,
                    )
                ),
                "training_time_sec": float(training_time_sec),
                "posterior_sampling_time_sec": float(posterior_sampling_time_sec),
                "diagnostics_time_sec": float(diagnostics_time_sec),
                "total_runtime_sec": float(total_runtime_sec),
                "reject_outside_prior": bool(reject_outside_prior),
                "max_sampling_time": max_sampling_time,
                "num_sampling_failures": int(num_sampling_failures),
                "num_sampling_fallbacks": int(num_sampling_fallbacks),
                "fallback_sampling_used": bool(fallback_sampling_used),
                "tarp_atc": float(atc),
                "tarp_ks_pvalue": float(ks_pval),
                "num_tarp_eval": int(test_sample),
                "num_tarp_posterior_samples": int(num_posterior_samples),
                "num_sbc_eval": int(test_sample),
                "num_sbc_posterior_samples": int(num_posterior_samples),
                "posterior_samples_path": str(posterior_h5_path),
                "training_summary_path": str(training_summary_path),
                "tarp_plot_path": str(tarp_plot_path),
                "sbc_plot_path": str(sbc_plot_path),
                "diagnostics_arrays_path": str(diagnostics_arrays_path),
                "status": status,
            }
        except Exception as exc:  # pragma: no cover - defensive path for long runs
            total_end = time.perf_counter()
            summary = {
                "seed": int(seed),
                "problem": problem,
                "missingness": mechanism,
                "epsilon": epsilon,
                "method": "npe_learned_imputation",
                "dataset_path": str(dataset_path),
                "max_train_samples": max_train_samples,
                "max_val_samples": max_val_samples,
                "lambda_recon": float(imputer_cfg.get("lambda_recon", 1.0)),
                "imputer_type": _resolve_imputer_type(problem, imputer_cfg),
                "density_estimator": str(npe_cfg.get("density_estimator", "nsf")),
                "total_runtime_sec": float(total_end - total_start),
                "status": f"failed: {exc}",
            }

        with (seed_output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        all_results.append(summary)

    with (output_dir / "all_results.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nFinished all seeds. Results: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
