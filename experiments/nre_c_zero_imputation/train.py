"""Train the experimental OUP MCAR NRE-C zero-imputation baseline."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from gapsbi.checkpointing import DEFAULT_CHECKPOINT_NAME, save_model_checkpoint
from gapsbi.datasets import apply_train_val_sample_limits
from gapsbi.io import load_gapsbi_hdf5
from gapsbi.methods.sbi_nre import (
    FixedSplitNRE_C,
    load_nre_c_zero_imputation_checkpoint,
    prepare_zero_imputed_nre_input,
)
from gapsbi.preprocessing.scalers import (
    infer_theta_transform,
    infer_x_transform,
    make_scaled_theta_prior,
    scale_theta_train_val_test,
    fit_x_scaler_on_full_train,
    theta_scaling_metadata,
)
from gapsbi.utils.seeding import set_all_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-reference-evaluation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config.get("problem") != "oup" or config.get("imputation", {}).get("method") != "zero":
        raise ValueError("This focused experiment supports only OUP zero imputation.")
    dataset, dataset_metadata = load_gapsbi_hdf5(config["dataset_path"])
    dataset = apply_train_val_sample_limits(
        dataset, config.get("max_train_samples"), config.get("max_val_samples")
    )
    theta_train, theta_val, _, theta_scaler = scale_theta_train_val_test(
        dataset["train"]["theta"], dataset["val"]["theta"], dataset["test"]["theta"],
        infer_theta_transform("oup"),
    )
    x_scaler, x_scaling_metadata = fit_x_scaler_on_full_train(
        dataset["train"]["x_full"], infer_x_transform("oup")
    )
    x_train = prepare_zero_imputed_nre_input(
        x_obs=dataset["train"]["x_obs"], mask=dataset["train"]["mask"],
        x_scaler=x_scaler, transform=x_scaling_metadata["transform"],
    )
    x_val = prepare_zero_imputed_nre_input(
        x_obs=dataset["val"]["x_obs"], mask=dataset["val"]["mask"],
        x_scaler=x_scaler, transform=x_scaling_metadata["transform"],
    )
    nre_config = config["nre"]
    prior = make_scaled_theta_prior("oup", 2, theta_train)

    for seed in config["seeds"]:
        seed = int(seed)
        set_all_seeds(seed)
        output_dir = Path(config["output_dir"]) / f"seed_{seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        inference = FixedSplitNRE_C(
            prior=prior, classifier=nre_config["classifier"], device=nre_config["device"],
            show_progress_bars=False,
        )
        inference.append_simulations(torch.cat((theta_train, theta_val)), torch.cat((x_train, x_val)))
        inference.set_fixed_train_val_split(len(theta_train), len(theta_val))
        started = time.perf_counter()
        estimator = inference.train(
            num_classes=int(nre_config["num_classes"]), gamma=float(nre_config["gamma"]),
            training_batch_size=int(nre_config["training_batch_size"]), validation_fraction=0.1,
            stop_after_epochs=int(nre_config["stop_after_epochs"]),
            max_num_epochs=int(nre_config["max_num_epochs"]),
        )
        runtime = time.perf_counter() - started
        history = inference.summary
        training_loss = [float(value) for value in history["training_loss"]]
        validation_loss = [float(value) for value in history["validation_loss"]]
        if not training_loss or not np.isfinite(training_loss).all() or not np.isfinite(validation_loss).all():
            raise FloatingPointError("NRE-C losses are nonfinite.")
        with torch.no_grad():
            log_ratio_finite = torch.isfinite(
                estimator.unnormalized_log_ratio(theta_val[:8], x_val[:8])
            ).all().item()
        if not log_ratio_finite:
            raise FloatingPointError("NRE-C log ratios are nonfinite.")
        checkpoint_path = save_model_checkpoint(
            output_dir / DEFAULT_CHECKPOINT_NAME, method="nre_c_zero_imputation", problem="oup",
            seed=seed, config=config, dataset_path=Path(config["dataset_path"]),
            theta_scaler=theta_scaler,
            theta_scaling_metadata=theta_scaling_metadata(infer_theta_transform("oup")),
            x_scaler=x_scaler, x_scaling_metadata=x_scaling_metadata, theta_dim=2,
            x_dim=x_train.shape[1], density_estimator=estimator,
            extra={
                "config_path": str(args.config), "classifier": nre_config["classifier"],
                "nre_c_num_classes": int(nre_config["num_classes"]),
                "nre_c_gamma": float(nre_config["gamma"]), "imputation": "zero_scaled_space",
                "mask_in_classifier_input": False, "training_runtime_sec": runtime,
                "epochs_trained": len(training_loss), "final_training_loss": training_loss[-1],
                "final_validation_loss": validation_loss[-1],
                "best_validation_loss": min(validation_loss), "dataset_metadata": dataset_metadata,
            },
        )
        _, reloaded = load_nre_c_zero_imputation_checkpoint(checkpoint_path)
        with torch.no_grad():
            reloaded_finite = torch.isfinite(
                reloaded.unnormalized_log_ratio(theta_val[:8], x_val[:8])
            ).all().item()
        summary = {
            "method": "nre_c_zero_imputation", "seed": seed,
            "num_train_examples": len(theta_train), "num_val_examples": len(theta_val),
            "training_runtime_sec": runtime, "epochs_trained": len(training_loss),
            "final_training_loss": training_loss[-1], "final_validation_loss": validation_loss[-1],
            "best_validation_loss": min(validation_loss), "nre_c_num_classes": int(nre_config["num_classes"]),
            "nre_c_gamma": float(nre_config["gamma"]), "imputation": "zero_scaled_space",
            "mask_in_classifier_input": False, "validation_log_ratio_finite": bool(log_ratio_finite),
            "reloaded_log_ratio_finite": bool(reloaded_finite), "checkpoint_path": str(checkpoint_path),
        }
        (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, sort_keys=True))
        if not args.skip_reference_evaluation:
            from evaluate import evaluate_checkpoint
            evaluate_checkpoint(checkpoint_path, config)


if __name__ == "__main__":
    main()
