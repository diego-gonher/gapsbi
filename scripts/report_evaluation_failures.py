"""Summarize numerical reliability from completed partial-reference analyses.

This report never recomputes posterior metrics. It reads the saved seed-level
partial-reference results, separates unavailable reference posteriors from
non-finite estimator evaluations, and writes seed-detail and grouped CSVs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CAMPAIGNS = (
    ("glu", "mcar"), ("glu", "mnar"),
    ("glm", "mcar"), ("glm", "mnar"),
    ("oup", "mcar"), ("oup", "mnar"),
    ("lotka_volterra", "mcar"), ("lotka_volterra", "mnar"),
)
METRICS = ("c2st", "mean_shift", "covariance_trace_ratio")


def expected_reference_count(problem: str, mechanism: str, fraction: float) -> int:
    """Return available partial-reference count without conflating estimator failure."""
    return 8 if (problem, mechanism, round(float(fraction), 2)) == (
        "lotka_volterra", "mnar", .50
    ) else 10


def load_seed_details(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    required = {
        "problem", "mechanism", "budget", "method", "fraction", "seed",
        "metric", "target", "n_valid_reference_observations", "source",
    }
    for problem, mechanism in CAMPAIGNS:
        path = root / f"{problem}_{mechanism}" / "seed_level_metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed partial-reference analysis: {path}")
        frame = pd.read_csv(path)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} lacks columns: {sorted(missing)}")
        frame = frame[(frame["target"] == "partial") & frame["metric"].isin(METRICS)].copy()
        identity = set(frame[["problem", "mechanism"]].drop_duplicates().itertuples(index=False, name=None))
        if identity != {(problem, mechanism)}:
            raise ValueError(f"Unexpected campaign identity in {path}: {identity}")
        frames.append(frame)

    rows = pd.concat(frames, ignore_index=True)
    keys = ["problem", "mechanism", "fraction", "method", "budget", "seed"]
    counts = rows.pivot(index=keys, columns="metric", values="n_valid_reference_observations")
    if counts.isna().any().any() or (counts.nunique(axis=1) != 1).any():
        raise ValueError("Metric rows disagree about valid-reference counts.")
    details = counts.iloc[:, :1].rename(columns={counts.columns[0]: "valid_reference_count"}).reset_index()
    source = rows.drop_duplicates(keys).set_index(keys)["source"]
    details["source"] = source.reindex(pd.MultiIndex.from_frame(details[keys])).to_numpy()
    details["available_reference_count"] = [
        expected_reference_count(row.problem, row.mechanism, row.fraction)
        for row in details.itertuples(index=False)
    ]
    details["missing_reference_posterior_count"] = 10 - details["available_reference_count"]
    details["nonfinite_estimator_reference_count"] = (
        details["available_reference_count"] - details["valid_reference_count"]
    )
    if (details["nonfinite_estimator_reference_count"] < 0).any():
        raise ValueError("Valid-reference count exceeds available partial references.")
    details["all_available_references_valid"] = details["nonfinite_estimator_reference_count"] == 0
    return details.sort_values(keys).reset_index(drop=True)


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    keys = ["problem", "mechanism", "fraction", "method", "budget"]
    summary = details.groupby(keys, as_index=False).agg(
        seed_count=("seed", "nunique"),
        available_reference_count=("available_reference_count", "max"),
        missing_reference_posterior_count=("missing_reference_posterior_count", "max"),
        valid_reference_count_total=("valid_reference_count", "sum"),
        nonfinite_estimator_reference_count_total=("nonfinite_estimator_reference_count", "sum"),
        seeds_with_nonfinite_estimator_evaluations=("nonfinite_estimator_reference_count", lambda x: int((x > 0).sum())),
        complete_seed_count=("all_available_references_valid", "sum"),
    )
    summary["expected_reference_evaluations_total"] = (
        summary["seed_count"] * summary["available_reference_count"]
    )
    return summary.sort_values(keys).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, default=Path("outputs/analysis/partial_reference_comparison"))
    parser.add_argument("--outdir", type=Path, default=Path("outputs/analysis/evaluation_reliability"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    outputs = (
        args.outdir / "evaluation_failure_seed_details.csv",
        args.outdir / "evaluation_failure_summary.csv",
        args.outdir / "evaluation_failure_manifest.json",
    )
    if any(path.exists() for path in outputs) and not args.overwrite:
        raise FileExistsError(f"{args.outdir} already contains a report; pass --overwrite to replace it.")
    details = load_seed_details(args.analysis_root)
    summary = summarize(details)
    args.outdir.mkdir(parents=True, exist_ok=True)
    details.to_csv(outputs[0], index=False)
    summary.to_csv(outputs[1], index=False)
    outputs[2].write_text(json.dumps({
        "schema": "gapsbi_evaluation_reliability_v1",
        "analysis_root": str(args.analysis_root),
        "seed_detail_path": str(outputs[0]),
        "summary_path": str(outputs[1]),
        "metric_rows_checked": list(METRICS),
        "failure_definition": "available_partial_references minus valid_reference_count",
        "missing_reference_definition": "canonical_partial_reference_count (10) minus available_partial_references",
        "sparse_reference_exception": "LV/MNAR at 50% has eight available references (original IDs 0,1,3,4,5,6,8,9)",
    }, indent=2), encoding="utf-8")
    print(f"seed_rows={len(details)} summary_rows={len(summary)} outdir={args.outdir}")


if __name__ == "__main__":
    main()
