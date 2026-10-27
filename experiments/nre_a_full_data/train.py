"""Train the experimental full-data OUP NRE-A baseline (no posterior evaluation)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from gapsbi.checkpointing import DEFAULT_CHECKPOINT_NAME, save_model_checkpoint
from gapsbi.datasets import apply_train_val_sample_limits
from gapsbi.io import load_gapsbi_hdf5
from gapsbi.methods.sbi_nre import FixedSplitNRE_A, load_nre_a_full_data_checkpoint
from gapsbi.preprocessing.scalers import (infer_theta_transform, infer_x_transform,
    make_scaled_theta_prior, scale_theta_train_val_test, scale_x_train_val_test,
    theta_scaling_metadata)
from gapsbi.utils.seeding import set_all_seeds


def _config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Config must be a mapping.")
    return value


def _history(inference: Any) -> tuple[list[float], list[float]]:
    summary = getattr(inference, "summary", getattr(inference, "_summary", {}))
    return ([float(x) for x in summary.get("training_loss", [])],
            [float(x) for x in summary.get("validation_loss", [])])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-reference-evaluation", action="store_true")
    args = parser.parse_args()
    config = _config(args.config)
    if config.get("problem") != "oup":
        raise ValueError("This minimal Step-1 trainer supports only full-data OUP.")
    dataset_path, output_dir = Path(config["dataset_path"]), Path(config["output_dir"])
    dataset, metadata = load_gapsbi_hdf5(dataset_path)
    dataset = apply_train_val_sample_limits(dataset, config["max_train_samples"], config["max_val_samples"])
    nre_cfg = config["nre"]
    theta_train, theta_val, _theta_test, theta_scaler = scale_theta_train_val_test(
        dataset["train"]["theta"], dataset["val"]["theta"], dataset["test"]["theta"], infer_theta_transform("oup"))
    x_train, x_val, _x_test, x_scaler, x_metadata = scale_x_train_val_test(
        dataset["train"]["x_full"], dataset["val"]["x_full"], dataset["test"]["x_full"], infer_x_transform("oup"))
    prior = make_scaled_theta_prior("oup", theta_train.shape[1], theta_train)
    for seed in [int(value) for value in config["seeds"]]:
        set_all_seeds(seed)
        directory = output_dir / f"seed_{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        inference = FixedSplitNRE_A(prior=prior, classifier=nre_cfg["classifier"], device=nre_cfg["device"], show_progress_bars=False)
        inference.append_simulations(torch.cat((theta_train, theta_val)), torch.cat((x_train, x_val)))
        inference.set_fixed_train_val_split(len(theta_train), len(theta_val))
        started = time.perf_counter()
        estimator = inference.train(training_batch_size=int(nre_cfg["training_batch_size"]), validation_fraction=.1,
            stop_after_epochs=int(nre_cfg["stop_after_epochs"]), max_num_epochs=int(nre_cfg["max_num_epochs"]))
        runtime = time.perf_counter() - started
        train_loss, val_loss = _history(inference)
        if not train_loss or not val_loss or not np.isfinite(train_loss).all() or not np.isfinite(val_loss).all():
            raise FloatingPointError("NRE-A training produced missing or nonfinite losses.")
        with torch.no_grad():
            log_ratio = estimator.unnormalized_log_ratio(theta_val[:8], x_val[:8])
        if not torch.isfinite(log_ratio).all():
            raise FloatingPointError("NRE-A validation log-ratios are nonfinite.")
        checkpoint = save_model_checkpoint(directory / DEFAULT_CHECKPOINT_NAME, method="nre_a_full_data",
            problem="oup", seed=seed, config=config, dataset_path=dataset_path, theta_scaler=theta_scaler,
            theta_scaling_metadata=theta_scaling_metadata(infer_theta_transform("oup")), x_scaler=x_scaler,
            x_scaling_metadata=x_metadata, theta_dim=theta_train.shape[1], x_dim=x_train.shape[1], density_estimator=estimator,
            extra={"config_path": str(args.config), "classifier": nre_cfg["classifier"], "training_runtime_sec": runtime,
                   "epochs_trained": len(train_loss), "final_training_loss": train_loss[-1], "final_validation_loss": val_loss[-1],
                   "best_validation_loss": min(val_loss), "dataset_metadata": metadata})
        _checkpoint, reloaded = load_nre_a_full_data_checkpoint(checkpoint)
        with torch.no_grad():
            reloaded_log_ratio = reloaded.unnormalized_log_ratio(theta_val[:8], x_val[:8])
        if not torch.isfinite(reloaded_log_ratio).all():
            raise FloatingPointError("Reloaded NRE-A checkpoint produced nonfinite log-ratios.")
        summary = {"method": "nre_a_full_data", "problem": "oup", "seed": seed, "num_train_examples": len(theta_train),
            "num_val_examples": len(theta_val), "training_runtime_sec": runtime, "epochs_trained": len(train_loss),
            "final_training_loss": train_loss[-1], "final_validation_loss": val_loss[-1], "best_validation_loss": min(val_loss),
            "validation_log_ratio_finite": True, "reloaded_log_ratio_finite": True,
            "checkpoint_path": str(checkpoint)}
        (directory / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, sort_keys=True))
        if not args.skip_reference_evaluation:
            # Local import keeps Step-1 training lightweight until a config opts
            # into the OUP-only grid/reference workflow.
            from evaluate import evaluate_checkpoint
            evaluate_checkpoint(checkpoint, config)


if __name__ == "__main__":
    main()
