#!/usr/bin/env python
"""Sequential, resumable launcher for the OUP NRE-C MCAR-10% campaign."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from gapsbi.evaluation.nre_c_campaign import validate_nre_c_campaign_run


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "experiments/nre_c_oup_mcar_sweep.yaml"
CAMPAIGN = ROOT / "logs/nre_c_oup_mcar_10pct"
METHODS = {
    "nre_c_zero_imputation": "zero_imputation",
    "nre_c_mask_augmentation": "mask_augmentation",
    "nre_c_learned_constant_imputation": "learned_constant_imputation",
    "nre_c_probabilistic_learned_imputation": "probabilistic_learned_imputation",
    "nre_c_masked_transformer_embedding": "masked_transformer_embedding",
}
TRAIN_NAMES = {name: f"experiments/nre_c_{folder}/train.py" for name, folder in METHODS.items()}
EVALUATE_NAMES = {name: f"experiments/nre_c_{folder}/evaluate.py" for name, folder in METHODS.items()}
OMIT_FROM_TRAINING_COMPARISON = {"output_dir", "evaluation", "seeds"}


def load_matrix(path: Path = MATRIX) -> dict[str, Any]:
    matrix = yaml.safe_load(path.read_text())
    if (matrix["problem"], matrix["mechanism"], float(matrix["missing_fraction"])) != ("oup", "mcar", .1):
        raise ValueError("This launcher is restricted to OUP/MCAR/10%.")
    return matrix


def jobs(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for method in matrix["methods"]:
        for budget, budget_config in matrix["budgets"].items():
            for seed in matrix["seeds"]:
                result.append({"method": method, "budget": budget, "seed": int(seed), "budget_config": budget_config})
    ids = [(j["method"], j["budget"], j["seed"]) for j in result]
    if len(result) != 75 or len(set(ids)) != 75:
        raise ValueError(f"Expected 75 unique configurations, found {len(result)} / {len(set(ids))}.")
    return result


def job_id(job: dict[str, Any]) -> str:
    return f"{job['method']}__{job['budget']}__seed{job['seed']}"


def run_dir(job: dict[str, Any], matrix: dict[str, Any]) -> Path:
    return ROOT / f"outputs_{job['budget']}_sim_budget" / job["method"] / "oup/mcar/oup_mcar_eps010" / f"seed_{job['seed']}"


def materialize_config(job: dict[str, Any], matrix: dict[str, Any]) -> dict[str, Any]:
    method = job["method"]
    template_path = ROOT / "experiments" / f"nre_c_{METHODS[method]}" / "low_sim_budget/oup_mcar_eps010_config.yaml"
    config = yaml.safe_load(template_path.read_text())
    budget = job["budget_config"]
    config.update({
        "dataset_path": matrix["dataset_path"],
        "output_dir": f"outputs_{job['budget']}_sim_budget/{method}/oup/mcar/oup_mcar_eps010",
        "seeds": [job["seed"]],
        "max_train_samples": budget["max_train_samples"],
        "max_val_samples": budget["max_val_samples"],
    })
    if "nre" in config:
        config["nre"]["training_batch_size"] = int(budget["training_batch_size"])
    if "rise" in config:
        config["rise"]["batch_size"] = int(budget["training_batch_size"])
    config["evaluation"]["partial_reference_path"] = matrix["partial_reference_path"]
    config["evaluation"]["reference_indices"] = [int(i) for i in matrix["reference_indices"]]
    return config


def _training_projection(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key not in OMIT_FROM_TRAINING_COMPARISON}


def check_checkpoint_compatibility(path: Path, config: dict[str, Any], method: str, seed: int) -> tuple[bool, str]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        return False, f"checkpoint is unreadable: {error}"
    if checkpoint.get("method") != method or int(checkpoint.get("seed", -1)) != int(seed):
        return False, "checkpoint method or seed does not match the requested run"
    if checkpoint.get("problem") != "oup":
        return False, "checkpoint problem is not OUP"
    if not isinstance(checkpoint.get("density_estimator_state_dict"), dict) or not checkpoint["density_estimator_state_dict"]:
        return False, "checkpoint has no estimator state"
    saved = checkpoint.get("config")
    if not isinstance(saved, dict):
        return False, "checkpoint has no training configuration"
    expected = _training_projection(config)
    actual = _training_projection(saved)
    if actual != expected:
        differing = sorted(key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key))
        return False, f"checkpoint training provenance differs in: {', '.join(differing)}"
    expected_path = Path(config["dataset_path"])
    saved_path = Path(str(checkpoint.get("dataset_path", "")))
    if saved_path.as_posix() != expected_path.as_posix():
        return False, "checkpoint dataset path does not match the campaign"
    return True, "checkpoint matches method, seed, dataset, budget, and training settings"


def current_status(job: dict[str, Any], matrix: dict[str, Any], prior: dict[str, Any] | None = None) -> dict[str, str]:
    config = materialize_config(job, matrix)
    folder = run_dir(job, matrix)
    validation = validate_nre_c_campaign_run(
        folder, expected_method=job["method"], expected_seed=job["seed"],
        reference_path=ROOT / matrix["partial_reference_path"],
    )
    checkpoint_path = folder / "model_checkpoint.pt"
    if checkpoint_path.exists():
        valid, reason = check_checkpoint_compatibility(checkpoint_path, config, job["method"], job["seed"])
        if not valid:
            return {"status": "failed", "reason": reason}
        if validation["status"] == "complete":
            return {"status": "complete", "reason": validation["reason"]}
        if validation["status"] == "failed":
            return validation
        return {"status": "incomplete", "reason": validation["reason"] + "; matching checkpoint can be reused"}
    if validation["status"] == "complete":
        return {"status": "failed", "reason": "evaluation is marked complete but its checkpoint is missing"}
    if validation["status"] == "failed":
        return validation
    if any((folder / name).exists() for name in ("reference_evaluation.json", "reference_posterior_samples.h5")):
        return {"status": "failed", "reason": "evaluation artifacts exist without a checkpoint; refusing to overwrite"}
    return {"status": "pending", "reason": "no checkpoint or evaluation artifacts yet"}


def commands_for(job: dict[str, Any], matrix: dict[str, Any], config_path: Path) -> tuple[list[str] | None, list[str]]:
    folder = run_dir(job, matrix)
    checkpoint = folder / "model_checkpoint.pt"
    evaluation = ["conda", "run", "--no-capture-output", "-n", "gapsbi_env", "python", EVALUATE_NAMES[job["method"]], "--checkpoint", str(checkpoint), "--config", str(config_path)]
    if not checkpoint.exists():
        training = ["conda", "run", "--no-capture-output", "-n", "gapsbi_env", "python", TRAIN_NAMES[job["method"]], "--config", str(config_path), "--skip-reference-evaluation"]
    else:
        training = None
    return training, evaluation


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _summary(matrix: dict[str, Any], records: dict[str, Any], *, refresh: bool = True) -> dict[str, Any]:
    counts = {name: 0 for name in ("complete", "incomplete", "failed", "pending")}
    rows = []
    for job in jobs(matrix):
        key = job_id(job)
        status = current_status(job, matrix, records.get(key)) if refresh or key not in records else records[key]
        if status["status"] == "pending" and records.get(key, {}).get("status") in {"failed", "incomplete"}:
            status = records[key]
        counts[status["status"]] += 1
        rows.append({"configuration": key, **status})
    return {"updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "counts": counts, "configurations": rows}


def _write_summary(matrix: dict[str, Any], records: dict[str, Any]) -> dict[str, Any]:
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    summary = _summary(matrix, records, refresh=False)
    (CAMPAIGN / "progress.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _acquire_lock() -> Path:
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    lock = CAMPAIGN / "sweep.lock"
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "w") as stream:
                stream.write(json.dumps({"pid": os.getpid(), "started_at": time.time()}))
            return lock
        except FileExistsError:
            data = _read_json(lock, {})
            pid = int(data.get("pid", -1))
            if pid <= 1:
                if time.time() - lock.stat().st_mtime > 10:
                    lock.unlink(missing_ok=True)
                    continue
                raise RuntimeError(f"Sweep lock is being initialized: {lock}")
            try:
                os.kill(pid, 0)
            except (ProcessLookupError, ValueError):
                lock.unlink(missing_ok=True)
                continue
            except PermissionError:
                pass
            raise RuntimeError(f"Sweep already active (lock {lock}, pid {pid}).")
    raise RuntimeError(f"Could not acquire campaign lock: {lock}")


def _run_job(job: dict[str, Any], matrix: dict[str, Any], records: dict[str, Any]) -> None:
    key = job_id(job)
    config = materialize_config(job, matrix)
    folder = run_dir(job, matrix)
    checkpoint = folder / "model_checkpoint.pt"
    initial = current_status(job, matrix, records.get(key))
    if initial["status"] == "complete":
        records[key] = {"status": "complete", "reason": "validated before launch"}
        return
    if initial["status"] == "failed":
        records[key] = {"status": "failed", "reason": initial["reason"]}
        return
    config_dir = CAMPAIGN / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{key}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    log_path = CAMPAIGN / "logs" / f"{key}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_exists = checkpoint.exists()
    if checkpoint_exists:
        valid, reason = check_checkpoint_compatibility(checkpoint, config, job["method"], job["seed"])
        if not valid:
            records[key] = {"status": "failed", "reason": reason, "updated_at": time.time()}
            return
    train_command, evaluate_command = commands_for(job, matrix, config_path)
    with log_path.open("a", buffering=1) as log:
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {key} ===\n")
        if train_command:
            log.write("TRAIN: " + " ".join(train_command) + "\n")
            code = subprocess.run(train_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
            if code:
                records[key] = {"status": "failed", "reason": f"training exited with code {code}", "training_exit_code": code, "updated_at": time.time()}
                _write_summary(matrix, records)
                return
            if not checkpoint.exists():
                records[key] = {"status": "failed", "reason": "training returned success but did not create checkpoint", "updated_at": time.time()}
                _write_summary(matrix, records)
                return
            valid, reason = check_checkpoint_compatibility(checkpoint, config, job["method"], job["seed"])
            if not valid:
                records[key] = {"status": "failed", "reason": "new checkpoint validation failed: " + reason, "updated_at": time.time()}
                _write_summary(matrix, records)
                return
        log.write("EVALUATE: " + " ".join(evaluate_command) + "\n")
        code = subprocess.run(evaluate_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
    result = validate_nre_c_campaign_run(folder, expected_method=job["method"], expected_seed=job["seed"], reference_path=ROOT / matrix["partial_reference_path"])
    if code:
        records[key] = {"status": "failed", "reason": f"evaluation exited with code {code}; {result['reason']}", "evaluation_exit_code": code, "updated_at": time.time()}
    elif result["status"] == "complete":
        records[key] = {"status": "complete", "reason": result["reason"], "updated_at": time.time()}
    else:
        records[key] = {"status": "failed", "reason": "evaluation did not produce a complete validated result: " + result["reason"], "updated_at": time.time()}
    _write_summary(matrix, records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Show all 75 configurations and planned commands without executing them.")
    parser.add_argument("--progress", action="store_true", help="Print the latest live completion summary.")
    parser.add_argument("--failures", action="store_true", help="List failed configurations and reasons.")
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    args = parser.parse_args()
    matrix = load_matrix(args.matrix)
    records_path = CAMPAIGN / "run_state.json"
    records = _read_json(records_path, {})
    if args.progress or args.failures:
        summary = _summary(matrix, records, refresh=True)
        if args.failures:
            print(json.dumps([row for row in summary["configurations"] if row["status"] == "failed"], indent=2))
        else:
            print(json.dumps(summary["counts"], sort_keys=True))
        return 0
    if args.dry_run:
        print(f"Campaign: {len(jobs(matrix))} unique configurations; no commands will run.")
        for job in jobs(matrix):
            status = current_status(job, matrix, records.get(job_id(job)))
            cfg_path = CAMPAIGN / "configs" / f"{job_id(job)}.yaml"
            train, evaluate = commands_for(job, matrix, cfg_path)
            print(f"[{status['status']}] {job_id(job)} — {status['reason']}")
            if status["status"] in ("complete", "failed"):
                continue
            print("  MATERIALIZE: " + str(cfg_path))
            if train:
                print("  TRAIN: " + " ".join(train))
            print("  EVALUATE: " + " ".join(evaluate))
        return 0
    lock = _acquire_lock()
    failures_path = CAMPAIGN / "failures.jsonl"
    try:
        # Take one artifact-based status snapshot; subsequent progress updates use
        # these records rather than repeatedly deserializing every checkpoint.
        for job in jobs(matrix):
            key = job_id(job)
            actual = current_status(job, matrix, records.get(key))
            if actual["status"] == "pending" and records.get(key, {}).get("status") == "failed":
                actual = records[key]
            records[key] = {**actual, "updated_at": time.time()}
        records_path.write_text(json.dumps(records, indent=2) + "\n")
        _write_summary(matrix, records)
        for index, job in enumerate(jobs(matrix), 1):
            print(f"[{index}/75] {job_id(job)}", flush=True)
            records[job_id(job)] = {"status": "incomplete", "reason": "configuration is currently executing", "updated_at": time.time()}
            records_path.write_text(json.dumps(records, indent=2) + "\n")
            _write_summary(matrix, records)
            _run_job(job, matrix, records)
            state = current_status(job, matrix, records.get(job_id(job)))
            recorded = records.get(job_id(job), {})
            if state["status"] == "failed" or recorded.get("status") == "failed":
                failure = recorded if recorded.get("status") == "failed" else state
                with failures_path.open("a") as stream:
                    stream.write(json.dumps({"configuration": job_id(job), **failure, "time": time.time()}) + "\n")
            records_path.parent.mkdir(parents=True, exist_ok=True)
            records_path.write_text(json.dumps(records, indent=2) + "\n")
            _write_summary(matrix, records)
    finally:
        lock.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
