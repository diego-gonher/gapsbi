"""Completion validation for the OUP NRE-C partial-reference campaign."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


EXPECTED_IDS = set(range(10))
REQUIRED_METRICS = {"c2st_accuracy", "posterior_mean_shift", "covariance_trace_ratio"}


def validate_nre_c_campaign_run(
    run_dir: str | Path, *, expected_method: str | None = None,
    expected_seed: int | None = None, reference_path: str | Path | None = None,
) -> dict[str, Any]:
    """Classify one run as complete, incomplete, or failed without mutation."""
    run = Path(run_dir)
    checkpoint_path, result_path, samples_path = (run / "model_checkpoint.pt", run / "reference_evaluation.json", run / "reference_posterior_samples.h5")
    if not checkpoint_path.exists() or not result_path.exists() or not samples_path.exists():
        return _status("incomplete", "missing checkpoint or reference-evaluation artifact")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        result = json.loads(result_path.read_text())
    except Exception as error:
        return _status("failed", f"unreadable checkpoint or JSON: {error}")
    if expected_method and checkpoint.get("method") != expected_method:
        return _status("failed", "checkpoint method mismatch")
    if expected_seed is not None and int(checkpoint.get("seed", -1)) != int(expected_seed):
        return _status("failed", "checkpoint seed mismatch")
    if result.get("method") != checkpoint.get("method") or int(result.get("seed", -1)) != int(checkpoint.get("seed", -2)):
        return _status("failed", "checkpoint/evaluation provenance mismatch")
    if result.get("posterior_target") != "theta_given_x_obs_mask":
        return _status("failed", "wrong posterior target")
    ids = {int(value) for value in result.get("reference_ids", [])}
    if ids != EXPECTED_IDS:
        return _status("incomplete", f"expected IDs 0-9, found {sorted(ids)}")
    rows = result.get("reference_metrics", [])
    if {int(row.get("reference_index", -1)) for row in rows} != EXPECTED_IDS:
        return _status("incomplete", "missing per-reference metrics")
    if not result.get("grid_convergence") or "selected_grid_resolution" not in result:
        return _status("incomplete", "missing grid diagnostics")
    for row in rows:
        if not REQUIRED_METRICS.issubset(row) or not all(np.isfinite(float(row[key])) for key in REQUIRED_METRICS):
            return _status("failed", "nonfinite or missing reference metric")
    try:
        with h5py.File(samples_path, "r") as h5:
            if "theta_posterior" not in h5 or h5["theta_posterior"].shape[0] != 10:
                return _status("incomplete", "missing ten-reference posterior samples")
            saved_ids = set(map(int, h5["reference_indices"][...]))
            if saved_ids != EXPECTED_IDS:
                return _status("incomplete", "saved reference IDs are incomplete")
            if not np.isfinite(h5["theta_posterior"][...]).all():
                return _status("failed", "nonfinite posterior samples")
            if reference_path is not None:
                with h5py.File(reference_path, "r") as ref:
                    for key in ("x_obs", "mask"):
                        if not np.array_equal(h5[key][...], ref[f"observations/{key}"][...]):
                            return _status("failed", f"saved {key} differs from reference artifact")
    except Exception as error:
        return _status("failed", f"invalid HDF5 output: {error}")
    return _status("complete", "all ten references and required artifacts validated")


def _status(status: str, reason: str) -> dict[str, str]:
    return {"status": status, "reason": reason}
