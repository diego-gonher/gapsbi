"""Train the experimental OUP MCAR NRE-C learned-constant baseline."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import yaml
from gapsbi.checkpointing import DEFAULT_CHECKPOINT_NAME, save_model_checkpoint
from gapsbi.datasets import apply_train_val_sample_limits
from gapsbi.io import load_gapsbi_hdf5
from gapsbi.methods.learned_imputation import prepare_learned_imputation_arrays
from gapsbi.methods.sbi_nre import (
    load_nre_c_learned_constant_imputation_checkpoint,
    train_learned_constant_imputation_nre_c,
)
from gapsbi.preprocessing.scalers import make_scaled_theta_prior
from gapsbi.utils.seeding import set_all_seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-reference-evaluation", action="store_true")
    args = parser.parse_args(); config = yaml.safe_load(args.config.read_text())
    if config.get("problem") != "oup":
        raise ValueError("This focused experiment supports OUP only.")
    dataset, dataset_metadata = load_gapsbi_hdf5(config["dataset_path"])
    dataset = apply_train_val_sample_limits(dataset, config.get("max_train_samples"), config.get("max_val_samples"))
    prepared = prepare_learned_imputation_arrays(dataset, "oup")
    nre, learned = config["nre"], config["learned_constant_imputation"]
    prior = make_scaled_theta_prior("oup", 2, prepared["theta_train"])
    for seed_value in config["seeds"]:
        seed = int(seed_value); set_all_seeds(seed)
        output_dir = Path(config["output_dir"]) / f"seed_{seed}"; output_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        result = train_learned_constant_imputation_nre_c(
            theta_train=prepared["theta_train"], x_obs_train_scaled=prepared["x_obs_train_scaled"], mask_train=prepared["mask_train"],
            theta_val=prepared["theta_val"], x_obs_val_scaled=prepared["x_obs_val_scaled"], mask_val=prepared["mask_val"], prior=prior,
            classifier=nre["classifier"], device=nre["device"], training_batch_size=int(nre["training_batch_size"]),
            num_classes=int(nre["num_classes"]), gamma=float(nre["gamma"]), learning_rate=float(nre["learning_rate"]),
            constant_learning_rate=float(learned["constant_learning_rate"]), stop_after_epochs=int(nre["stop_after_epochs"]),
            max_num_epochs=int(nre["max_num_epochs"]), init_value=float(learned["init_value"]),
        )
        runtime = time.perf_counter() - started
        initial_constants = np.full(prepared["x_obs_train_scaled"].shape[1], float(learned["init_value"]))
        constants = result.imputer.constants.detach().cpu().numpy()
        update_l2 = float(np.linalg.norm(constants - initial_constants))
        if not result.constant_gradient_observed or update_l2 == 0.0:
            raise RuntimeError("Learned constants did not receive a nonzero NRE-C gradient and update.")
        with torch.no_grad():
            finite = torch.isfinite(result.estimator.unnormalized_log_ratio(prepared["theta_val"][:8], result.imputer(prepared["x_obs_val_scaled"][:8], prepared["mask_val"][:8]))).all().item()
        if not finite: raise FloatingPointError("NRE-C log ratios are nonfinite.")
        checkpoint = save_model_checkpoint(
            output_dir / DEFAULT_CHECKPOINT_NAME, method="nre_c_learned_constant_imputation", problem="oup", seed=seed, config=config,
            dataset_path=Path(config["dataset_path"]), theta_scaler=prepared["theta_scaler"], theta_scaling_metadata=prepared["theta_scaling_metadata"],
            x_scaler=prepared["x_scaler"], x_scaling_metadata=prepared["x_scaling_metadata"], theta_dim=2, x_dim=constants.size,
            density_estimator=result.estimator, extra={"imputer_state_dict": result.imputer.state_dict(), "learned_constants_scaled": constants,
                "initial_constants_scaled": initial_constants, "learned_constants_update_l2": update_l2, "constant_gradient_observed": result.constant_gradient_observed,
                "nre_c_num_classes": int(nre["num_classes"]), "nre_c_gamma": float(nre["gamma"]), "training_runtime_sec": runtime,
                "epochs_trained": len(result.training_loss), "best_epoch": result.best_epoch, "best_validation_loss": min(result.validation_loss),
                "final_training_loss": result.training_loss[-1], "final_validation_loss": result.validation_loss[-1], "dataset_metadata": dataset_metadata})
        _, reloaded, reloaded_imputer = load_nre_c_learned_constant_imputation_checkpoint(checkpoint)
        with torch.no_grad():
            reloaded_finite = torch.isfinite(reloaded.unnormalized_log_ratio(prepared["theta_val"][:8], reloaded_imputer(prepared["x_obs_val_scaled"][:8], prepared["mask_val"][:8]))).all().item()
        summary = {"method": "nre_c_learned_constant_imputation", "seed": seed, "num_train_examples": len(prepared["theta_train"]),
                   "num_val_examples": len(prepared["theta_val"]), "training_runtime_sec": runtime, "epochs_trained": len(result.training_loss),
                   "best_epoch": result.best_epoch, "final_training_loss": result.training_loss[-1], "final_validation_loss": result.validation_loss[-1],
                   "best_validation_loss": min(result.validation_loss), "constant_gradient_observed": result.constant_gradient_observed,
                   "learned_constants_update_l2": update_l2, "validation_log_ratio_finite": bool(finite), "reloaded_log_ratio_finite": bool(reloaded_finite),
                   "checkpoint_path": str(checkpoint)}
        (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n"); print(json.dumps(summary, sort_keys=True))
        if not args.skip_reference_evaluation:
            from evaluate import evaluate_checkpoint
            evaluate_checkpoint(checkpoint, config)


if __name__ == "__main__": main()
