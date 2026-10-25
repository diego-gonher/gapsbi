"""Aggregate persisted, independently initialized LV/MCAR follow-up ensembles.

This intentionally treats one preselected walker from each *independent*
ensemble as a chain for ArviZ diagnostics. It never treats the interacting
walkers of an emcee ensemble as independent chains.
"""
from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=_json_default), encoding="utf-8")
    temporary.replace(path)


def _arviz(chains: list[np.ndarray], cache_dir: Path) -> dict[str, Any]:
    import platformdirs

    cache_dir.mkdir(parents=True, exist_ok=True)
    platformdirs.user_cache_dir = lambda *_args, **_kwargs: str(cache_dir)
    import arviz as az

    idata = az.from_dict(posterior={"log_theta": np.stack(chains, axis=0)})
    return {
        "method": "ArviZ rank-normalized diagnostics from fixed walker 0 of three independent ensembles",
        "rank_rhat": az.rhat(idata, method="rank")["log_theta"].values,
        "bulk_ess": az.ess(idata, method="bulk")["log_theta"].values,
        "tail_ess": az.ess(idata, method="tail")["log_theta"].values,
    }


def _summary(raw: np.ndarray) -> dict[str, Any]:
    return {
        "mean": raw.mean(axis=0),
        "covariance": np.cov(raw, rowvar=False),
        "quantiles_05_50_95": np.quantile(raw, [0.05, 0.5, 0.95], axis=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", required=True, type=Path,
                        help="Completed one-ensemble pilot JSON; repeat for two or more independent ensembles.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.result) < 2:
        raise ValueError("At least two independent ensemble result files are required.")

    records, representative, full_chains, log_probabilities = [], [], [], []
    for path in args.result:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "completed" or len(record.get("ensembles", [])) != 1:
            raise ValueError(f"{path} is not a completed single-ensemble run.")
        summary = record["ensembles"][0]
        chain_path = Path(summary["chain_file"])
        with np.load(chain_path) as arrays:
            log_chain = np.asarray(arrays["log_theta_chain"], dtype=float)
            log_prob = np.asarray(arrays["log_posterior_chain"], dtype=float)
        if log_chain.shape[:2] != log_prob.shape or log_chain.ndim != 3 or log_chain.shape[2] != 4:
            raise ValueError(f"Invalid chain shape in {chain_path}")
        records.append({"result_file": str(path), "chain_file": str(chain_path), "summary": summary,
                        "initialization": record.get("targeted_initialization")})
        representative.append(log_chain[:, 0, :])
        full_chains.append(log_chain.reshape(-1, 4))
        log_probabilities.append(log_prob.reshape(-1))

    agreements = []
    for (left_record, left), (right_record, right) in combinations(zip(records, full_chains), 2):
        ls, rs = _summary(np.exp(left)), _summary(np.exp(right))
        agreements.append({
            "results": [left_record["result_file"], right_record["result_file"]],
            "mean_l2_delta": float(np.linalg.norm(ls["mean"] - rs["mean"])),
            "covariance_fro_delta": float(np.linalg.norm(ls["covariance"] - rs["covariance"], ord="fro")),
            "max_quantile_abs_delta": float(np.max(np.abs(ls["quantiles_05_50_95"] - rs["quantiles_05_50_95"]))),
        })

    # The threshold alpha=1.1 is a transparent diagnostic boundary between the
    # old low region (~0.86) and prior high tail (>1.5); it is not a mode label
    # or a posterior mixture-weight estimate.
    high_threshold = float(np.log(1.1))
    region_rows = []
    for record, chain, log_prob in zip(records, full_chains, log_probabilities):
        high = chain[:, 0] > high_threshold
        # Count transitions in the original time-by-walker layout for a direct
        # mixing diagnostic rather than interpreting pooled occupancy as mass.
        with np.load(record["chain_file"]) as arrays:
            per_walker = np.asarray(arrays["log_theta_chain"], dtype=float)[:, :, 0] > high_threshold
        transitions = int(np.sum(per_walker[1:] != per_walker[:-1]))
        region_rows.append({
            "result_file": record["result_file"], "alpha_high_threshold": 1.1,
            "high_region_fraction_within_ensemble": float(np.mean(high)),
            "high_region_transition_count": transitions,
            "log_posterior_max": float(np.max(log_prob)),
            "log_posterior_high_region_max": float(np.max(log_prob[high])) if np.any(high) else None,
            "log_posterior_low_region_max": float(np.max(log_prob[~high])) if np.any(~high) else None,
            "near_maximum_log_density_window": 20.0,
            "near_maximum_fraction": float(np.mean(log_prob >= np.max(log_prob) - 20.0)),
            "high_region_near_maximum_fraction": float(np.mean(high & (log_prob >= np.max(log_prob) - 20.0))),
            "walkers_majority_high_region": int(np.sum(np.mean(per_walker, axis=0) > 0.5)),
        })

    output = {
        "schema": "gapsbi_lv_mcar_targeted_followup_v1",
        "records": records,
        "independent_representative_diagnostics": _arviz(representative, args.output.parent / ".arviz_cache"),
        "pairwise_pooled_ensemble_agreement": agreements,
        "per_ensemble_raw_parameter_summaries": [_summary(np.exp(chain)) for chain in full_chains],
        "high_alpha_region_diagnostic": region_rows,
        "caveat": "Occupancies are initialization-dependent until independent ensembles demonstrably mix; do not use them as posterior mode weights.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output, output)


if __name__ == "__main__":
    main()
