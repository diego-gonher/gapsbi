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
from sbi.utils import BoxUniform

from gapsbi.evaluation.posterior_sampling import sample_posteriors_once
from gapsbi.evaluation.sbc import compute_sbc_ranks_from_samples
from gapsbi.evaluation.tarp import compute_tarp_from_samples
from gapsbi.io import load_gapsbi_hdf5
from gapsbi.methods.sbi_npe import FixedSplitNPE_C
from gapsbi.preprocessing.scalers import (
    infer_x_transform,
    scale_theta_train_val_test,
    scale_x_train_val_test,
)
from gapsbi.utils.seeding import set_all_seeds

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NPE on full-data GAPSBI splits.")
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
    """Extract epochs and best validation loss with version-robust fallbacks."""
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    dataset_path = Path(config["dataset_path"])
    output_dir = Path(config["output_dir"])
    seeds = [int(s) for s in config["seeds"]]
    npe_cfg = config["npe"]
    eval_cfg = config["evaluation"]

    output_dir.mkdir(parents=True, exist_ok=True)

    problem = infer_problem_name(config.get("problem"), dataset_path)
    dataset, _metadata = load_gapsbi_hdf5(dataset_path)

    theta_train_np = dataset["train"]["theta"]
    theta_val_np = dataset["val"]["theta"]
    theta_test_np = dataset["test"]["theta"]

    x_train_np = dataset["train"]["x_full"]
    x_val_np = dataset["val"]["x_full"]
    x_test_np = dataset["test"]["x_full"]

    theta_train, theta_val, theta_test, theta_scaler = scale_theta_train_val_test(
        theta_train=theta_train_np,
        theta_val=theta_val_np,
        theta_test=theta_test_np,
    )

    x_transform = infer_x_transform(problem)
    x_train, x_val, x_test, x_scaler, x_scaling_metadata = scale_x_train_val_test(
        x_train=x_train_np,
        x_val=x_val_np,
        x_test=x_test_np,
        transform=x_transform,
    )

    theta_dim = theta_train.shape[1]
    x_dim = x_train.shape[1]
    num_train_examples = theta_train.shape[0]
    num_val_examples = theta_val.shape[0]
    num_test_examples = theta_test.shape[0]
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
        posterior_samples_scaled = sample_posteriors_once(
            posterior=posterior,
            x_eval=x_eval,
            num_posterior_samples=num_posterior_samples,
            seed=seed + 10_000,
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
            f.attrs["method"] = "npe_full_data"
            f.attrs["seed"] = int(seed)
            f.attrs["num_eval"] = int(num_eval)
            f.attrs["num_posterior_samples"] = int(num_samples)
            f.attrs["theta_dim"] = int(theta_dim_local)
            f.attrs["x_transform"] = x_scaling_metadata["transform"]

        # Time diagnostics: TARP/SBC computations and diagnostic artifacts.
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
            "method": "npe_full_data",
            "dataset_path": str(dataset_path),
            "x_transform": x_scaling_metadata["transform"],
            "training_time_sec": float(training_time_sec),
            "posterior_sampling_time_sec": float(posterior_sampling_time_sec),
            "epochs_trained": epochs_trained,
            "best_validation_loss": best_validation_loss,
            "num_train_examples": int(num_train_examples),
            "num_val_examples": int(num_val_examples),
            "num_test_examples": int(num_test_examples),
            "theta_dim": int(theta_dim),
            "x_dim": int(x_dim),
            "device": str(npe_cfg.get("device", "cpu")),
            "density_estimator": str(npe_cfg["density_estimator"]),
            "num_parameters": int(num_parameters),
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
        }
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
