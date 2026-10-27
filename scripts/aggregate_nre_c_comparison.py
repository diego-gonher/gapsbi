"""Aggregate saved OUP/NPE/NRE-C reference metrics; never reruns inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import yaml

from gapsbi.evaluation.nre_c_campaign import validate_nre_c_campaign_run


METRICS = {
    "c2st": ("c2st_accuracy", "mean"),
    "mean_shift": ("posterior_mean_shift", "median"),
    "covariance_trace_ratio": ("covariance_trace_ratio", "median"),
}
METHODS = {
    "zero_imputation": ("npe_zero_imputation", "nre_c_zero_imputation", "Zero Imputation"),
    "mask_augmentation": ("npe_mask_augmentation", "nre_c_mask_augmentation", "Zero + Mask"),
    "learned_constant_imputation": ("npe_learned_constant_imputation", "nre_c_learned_constant_imputation", "Learned Constant"),
    "probabilistic_learned_imputation": ("npe_probabilistic_learned_imputation", "nre_c_probabilistic_learned_imputation", "Probabilistic Learned Imputation"),
    "masked_transformer_embedding": ("npe_masked_transformer_embedding", "nre_c_masked_transformer_embedding", "Masked Transformer Embedding"),
}
EXPECTED_SEEDS = (101, 202, 303, 404, 505)
EXPECTED_BUDGETS = ("low", "mid", "high")
SEED_COLUMNS = [
    "problem", "mechanism", "budget", "method", "fraction", "seed", "metric", "target", "value",
    "configuration_complete", "n_valid_reference_observations", "reference_ids", "source", "backbone",
]
AGG_COLUMNS = [
    "problem", "mechanism", "fraction", "budget", "method", "backbone", "target", "metric",
    "mean", "std", "ci95_low", "ci95_high", "n_seeds", "source_aggregate",
]


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problem", default="oup")
    p.add_argument("--mechanism", default="mcar")
    p.add_argument("--fraction", type=float, default=.10)
    p.add_argument("--outputs-root", type=Path, default=Path("."))
    p.add_argument("--npe-analysis-dir", type=Path, default=None)
    p.add_argument("--analysis-dir", type=Path, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=list(EXPECTED_SEEDS))
    p.add_argument("--budgets", nargs="+", choices=EXPECTED_BUDGETS, default=list(EXPECTED_BUDGETS))
    p.add_argument("--reference-ids", type=int, nargs="+", default=list(range(10)))
    return p.parse_args()


def fraction_tag(fraction: float) -> str:
    return f"eps{round(fraction * 100):03d}"


def run_stem(problem: str, mechanism: str, fraction: float) -> str:
    tag = fraction_tag(fraction)
    if problem == "glm":
        return f"glm_raw_{mechanism}_{tag}"
    if problem == "lotka_volterra":
        family = "time_block_mcar" if mechanism == "mcar" else "log_total_mnar"
        return f"lotka_volterra_{family}_{tag}"
    return f"{problem}_{mechanism}_{tag}"


def resolve_path(root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def validate_ids(ids: list[int] | np.ndarray, expected: list[int], label: str) -> None:
    values = [int(x) for x in ids]
    if len(values) != len(set(values)) or set(values) != set(expected):
        raise ValueError(f"{label}: expected reference IDs {expected}, got {values}.")


def summarize_observations(rows: list[dict[str, Any]], metric: str, *, id_key: str = "reference_index") -> tuple[float, int]:
    field, reducer = METRICS[metric]
    values_by_id = {int(row[id_key]): float(row[field]) for row in rows}
    values = np.asarray(list(values_by_id.values()), dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size != values.size:
        raise ValueError(f"{metric} has nonfinite reference values.")
    return float(np.mean(finite) if reducer == "mean" else np.median(finite)), int(finite.size)


def aggregate_seed_values(values: list[float]) -> dict[str, float | int]:
    x = np.asarray(values, dtype=float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Seed aggregates require nonempty finite values.")
    n = len(x)
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    mean = float(np.mean(x))
    half = 1.96 * sd / math.sqrt(n)
    return {"mean": mean, "std": sd, "ci95_low": mean - half, "ci95_high": mean + half, "n_seeds": n}


def validate_configuration_grid(frame: pd.DataFrame, *, methods: list[str], budgets: list[str], seeds: list[int], metrics: list[str] | None = None) -> None:
    metrics = metrics or list(METRICS)
    expected = {(method, budget, int(seed), metric) for method in methods for budget in budgets for seed in seeds for metric in metrics}
    found = set(zip(frame.method, frame.budget, frame.seed.astype(int), frame.metric))
    if frame.duplicated(["method", "budget", "seed", "metric"]).any():
        raise ValueError("Duplicate method/budget/seed/metric rows.")
    missing, extra = expected - found, found - expected
    if missing or extra:
        raise ValueError(f"Configuration coverage mismatch: {len(missing)} missing, {len(extra)} unexpected; first missing={sorted(missing)[:1]}.")


def _npe_analysis_inputs(args_: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, Path, dict[str, Any]]:
    analysis = args_.npe_analysis_dir or Path("outputs/analysis/partial_reference_comparison") / f"{args_.problem}_{args_.mechanism}"
    seed_path, aggregate_path = analysis / "seed_level_metrics.csv", analysis / "full_vs_partial_summary.csv"
    if not seed_path.is_file() or not aggregate_path.is_file():
        raise FileNotFoundError(f"NPE partial-reference aggregates are missing under {analysis}.")
    seeds = pd.read_csv(seed_path)
    seeds = seeds[(seeds.target == "partial") & np.isclose(seeds.fraction, args_.fraction)].copy()
    methods = [pair[0] for pair in METHODS.values()]
    seeds = seeds[seeds.method.isin(methods) & seeds.budget.isin(args_.budgets) & seeds.seed.isin(args_.seeds)].copy()
    if (not seeds.configuration_complete.all()) or not (seeds.n_valid_reference_observations == len(args_.reference_ids)).all():
        raise ValueError("NPE partial seed rows include incomplete reference comparisons.")
    validate_configuration_grid(seeds.rename(columns={"fraction": "fraction"}), methods=methods, budgets=args_.budgets, seeds=args_.seeds)
    # Verify the saved per-reference rows and target rather than trusting only CSV summaries.
    common_metric_settings = None
    for row in seeds.itertuples(index=False):
        source = resolve_path(args_.outputs_root, row.source)
        data = json.loads(source.read_text())
        if (data.get("problem"), data.get("mechanism"), float(data.get("fraction", -1)), data.get("method"), data.get("budget"), int(data.get("seed", -1))) != (
            args_.problem, args_.mechanism, args_.fraction, row.method, row.budget, int(row.seed)
        ):
            raise ValueError(f"NPE source identity mismatch: {source}")
        if data.get("aggregate_policy") != "original_gapsbi_finite_observations_v1":
            raise ValueError(f"Unexpected NPE metric aggregation policy in {source}")
        settings = data.get("metric_settings", {})
        relevant_settings = {k: settings.get(k) for k in ("max_samples_per_observation", "c2st_max_samples_per_observation", "c2st_n_folds", "c2st_hidden_layer_scale", "c2st_max_iter")}
        if common_metric_settings is None:
            common_metric_settings = relevant_settings
        elif relevant_settings != common_metric_settings:
            raise ValueError(f"NPE reference metric settings differ between configurations: {source}")
        section = data.get("partial_reference", {})
        ref_rows = section.get("rows", [])
        validate_ids([r.get("reference_index", -1) for r in ref_rows], args_.reference_ids, str(source))
        refpath = data.get("provenance", {}).get("partial_reference_path")
        if not refpath:
            raise ValueError(f"NPE partial-reference provenance missing: {source}")
        if float(data.get("fraction", -1)) != args_.fraction:
            raise ValueError(f"NPE missingness fraction mismatch: {source}")
        metric_value, n_valid = summarize_observations(ref_rows, row.metric)
        if n_valid != int(row.n_valid_reference_observations) or not np.isclose(metric_value, float(row.value), rtol=0, atol=1e-12):
            raise ValueError(f"NPE saved seed aggregate differs from reference rows: {source} ({row.metric})")
        expected_ref = resolve_path(args_.outputs_root, refpath)
        if not expected_ref.is_file():
            raise FileNotFoundError(expected_ref)
    aggregate = pd.read_csv(aggregate_path)
    aggregate = aggregate[np.isclose(aggregate.fraction, args_.fraction) & aggregate.method.isin(methods) & aggregate.budget.isin(args_.budgets)].copy()
    _validate_npe_aggregates(seeds, aggregate, methods, args_.budgets, args_.seeds)
    return seeds, aggregate, analysis, common_metric_settings or {}


def _validate_npe_aggregates(seeds: pd.DataFrame, aggregate: pd.DataFrame, methods: list[str], budgets: list[str], seeds_expected: list[int]) -> None:
    for method in methods:
        for budget in budgets:
            for metric in METRICS:
                group = seeds[(seeds.method == method) & (seeds.budget == budget) & (seeds.metric == metric)]
                summary = aggregate[(aggregate.method == method) & (aggregate.budget == budget) & (aggregate.metric == metric)]
                if len(group) != len(seeds_expected) or len(summary) != 1:
                    raise ValueError(f"Missing NPE aggregate for {method}/{budget}/{metric}.")
                calculated = aggregate_seed_values(group.sort_values("seed").value.astype(float).tolist())
                row = summary.iloc[0]
                if int(row.n_seeds_partial) != len(seeds_expected) or int(row.n_complete_configurations_partial) != len(seeds_expected):
                    raise ValueError(f"NPE aggregate has incomplete seed count: {method}/{budget}/{metric}.")
                if not np.isclose(row.mean_partial, calculated["mean"], rtol=0, atol=1e-12) or not np.isclose(row.std_partial, calculated["std"], rtol=0, atol=1e-12):
                    raise ValueError(f"NPE original aggregate does not reproduce from seed CSV: {method}/{budget}/{metric}.")


def _validate_reference_file(reference_path: Path, observations_path: Path, expected_ids: list[int], keys: tuple[str, ...]) -> None:
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    with h5py.File(reference_path, "r") as ref, h5py.File(observations_path, "r") as saved:
        for key in keys:
            ref_key = f"observations/{key}"
            if ref_key not in ref or key not in saved or not np.array_equal(ref[ref_key][...], saved[key][...]):
                raise ValueError(f"Saved full-data baseline {key} does not align with {reference_path}.")
        if "theta_posterior" not in saved or saved["theta_posterior"].shape[0] != len(expected_ids):
            raise ValueError(f"Full-data reference sample shape is incomplete: {observations_path}")


def _load_partial_nre_c(args_: argparse.Namespace, reference_path: Path, npe_metric_settings: dict[str, Any]) -> pd.DataFrame:
    root = args_.outputs_root
    stem = run_stem(args_.problem, args_.mechanism, args_.fraction)
    rows = []
    for semantic, (_npe, nrec, _label) in METHODS.items():
        config_path = Path(__file__).resolve().parents[1] / "experiments" / nrec / "low_sim_budget" / f"{run_stem(args_.problem, args_.mechanism, args_.fraction)}_config.yaml"
        config_metrics = yaml.safe_load(config_path.read_text()).get("evaluation", {}).get("reference_metrics", {}) if config_path.is_file() else {}
        for config_key, npe_key in (("c2st_max_samples_per_observation", "c2st_max_samples_per_observation"), ("c2st_n_folds", "c2st_n_folds"), ("c2st_hidden_layer_scale", "c2st_hidden_layer_scale"), ("c2st_max_iter", "c2st_max_iter")):
            if config_metrics and int(config_metrics[config_key]) != int(npe_metric_settings[npe_key]):
                raise ValueError(f"NPE/NRE-C C2ST settings differ for {nrec}: {config_key}")
        for budget in args_.budgets:
            for seed in args_.seeds:
                run = root / f"outputs_{budget}_sim_budget" / nrec / args_.problem / args_.mechanism / stem / f"seed_{seed}"
                status = validate_nre_c_campaign_run(run, expected_method=nrec, expected_seed=seed, reference_path=reference_path)
                if status["status"] != "complete":
                    raise ValueError(f"NRE-C run is not complete: {run}: {status['status']} ({status['reason']})")
                result_path = run / "reference_evaluation.json"
                result = json.loads(result_path.read_text())
                if result.get("posterior_target") != "theta_given_x_obs_mask":
                    raise ValueError(f"NRE-C partial target mismatch: {result_path}")
                if resolve_path(root, result.get("reference_artifact", "")) != reference_path:
                    raise ValueError(f"NRE-C reference artifact path mismatch: {result_path}")
                metadata = result.get("reference_metadata", {})
                mask = metadata.get("mask", {})
                if (metadata.get("problem"), metadata.get("missingness_mechanism"), float(mask.get("missing_fraction", -1))) != (args_.problem, args_.mechanism, args_.fraction):
                    raise ValueError(f"NRE-C dataset/missingness provenance mismatch: {result_path}")
                validate_ids(result.get("reference_ids", []), args_.reference_ids, str(result_path))
                metric_rows = result.get("reference_metrics", [])
                validate_ids([r.get("reference_index", -1) for r in metric_rows], args_.reference_ids, f"metrics in {result_path}")
                if not all(r.get("reference_metrics_valid", True) for r in metric_rows):
                    raise ValueError(f"NRE-C contains invalid reference metric rows: {result_path}")
                if any(int(r.get("num_samples_used", -1)) != int(npe_metric_settings["max_samples_per_observation"]) for r in metric_rows):
                    raise ValueError(f"NPE/NRE-C reference sample counts differ: {result_path}")
                for key in ("c2st_num_samples_used", "c2st_n_folds", "c2st_hidden_layer_scale"):
                    expected = npe_metric_settings[{"c2st_num_samples_used":"c2st_max_samples_per_observation","c2st_n_folds":"c2st_n_folds","c2st_hidden_layer_scale":"c2st_hidden_layer_scale"}[key]]
                    if any(int(r.get(key, -1)) != int(expected) for r in metric_rows):
                        raise ValueError(f"NPE/NRE-C C2ST setting {key} differs: {result_path}")
                for metric, (field, _) in METRICS.items():
                    value, n_valid = summarize_observations(metric_rows, metric)
                    saved_aggregate = result.get("reference_metric_aggregate", {}).get(field)
                    if saved_aggregate is None or not np.isclose(value, float(saved_aggregate), rtol=0, atol=1e-12):
                        raise ValueError(f"NRE-C stored aggregate disagrees with per-reference rows: {result_path}/{metric}")
                    rows.append({"problem": args_.problem, "mechanism": args_.mechanism, "budget": budget,
                        "method": nrec, "fraction": args_.fraction, "seed": seed, "metric": metric,
                        "target": "partial", "value": value, "configuration_complete": True,
                        "n_valid_reference_observations": n_valid, "reference_ids": json.dumps(args_.reference_ids),
                        "source": str(result_path), "backbone": "NRE-C"})
    frame = pd.DataFrame(rows, columns=SEED_COLUMNS)
    validate_configuration_grid(frame, methods=[x[1] for x in METHODS.values()], budgets=args_.budgets, seeds=args_.seeds)
    return frame


def _load_full_data_baselines(args_: argparse.Namespace, full_reference_path: Path, npe_baseline_csv: Path) -> pd.DataFrame:
    """Load only existing full-data baselines; all their targets are p(theta|x_full)."""
    root = args_.outputs_root
    npe_baselines = pd.read_csv(npe_baseline_csv)
    npe_baselines = npe_baselines[(npe_baselines.problem == args_.problem) & (npe_baselines.method == "npe_full_data") & npe_baselines.budget.isin([f"{b}_sim_budget" for b in args_.budgets])]
    rows = []
    for budget in args_.budgets:
        budget_dir = root / f"outputs_{budget}_sim_budget"
        npe_path = budget_dir / "npe_full_data" / f"{args_.problem}_full_data" / "all_results.json"
        if npe_path.is_file():
            runs = json.loads(npe_path.read_text())
            if {int(x["seed"]) for x in runs} != set(args_.seeds):
                raise ValueError(f"NPE full-data baseline seed coverage mismatch: {npe_path}")
            for run in runs:
                seed = int(run["seed"])
                if int(run.get("num_reference_eval", -1)) != len(args_.reference_ids):
                    raise ValueError(f"NPE full-data baseline has wrong reference count: {npe_path}/{seed}")
                if resolve_path(root, run["reference_path"]) != full_reference_path:
                    raise ValueError(f"NPE full-data reference path mismatch: {npe_path}/{seed}")
                matching = npe_baselines[(npe_baselines.budget == f"{budget}_sim_budget") & (npe_baselines.seed == seed)]
                if len(matching) != 1:
                    raise ValueError(f"Missing original NPE full-data aggregate for {budget}/{seed}.")
                original = matching.iloc[0]
                arrays = {"c2st": run["reference_c2st_accuracy"], "mean_shift": run["reference_posterior_mean_shift"], "covariance_trace_ratio": run["reference_covariance_trace_ratio"]}
                for metric, values in arrays.items():
                    if len(values) != len(args_.reference_ids) or not np.isfinite(np.asarray(values, dtype=float)).all():
                        raise ValueError(f"NPE full-data baseline metrics are incomplete: {npe_path}/{seed}/{metric}")
                    value = float(np.mean(values) if METRICS[metric][1] == "mean" else np.median(values))
                    original_column = {"c2st":"reference_c2st_accuracy_mean", "mean_shift":"reference_posterior_mean_shift_median", "covariance_trace_ratio":"reference_covariance_trace_ratio_median"}[metric]
                    if not np.isclose(value, float(original[original_column]), rtol=0, atol=1e-12):
                        raise ValueError(f"NPE full-data saved aggregate mismatch: {npe_path}/{seed}/{metric}")
                    rows.append({"problem":args_.problem,"mechanism":"none","budget":budget,"method":"npe_full_data","fraction":0.,"seed":seed,"metric":metric,"target":"full","value":value,"configuration_complete":True,"n_valid_reference_observations":len(values),"reference_ids":json.dumps(args_.reference_ids),"source":str(npe_path),"backbone":"NPE"})
        # NRE-C full-data baseline is included only when the saved complete matrix exists.
        nrec_dir = budget_dir / "nre_c_full_data" / f"{args_.problem}_full_data"
        nrec_paths = [nrec_dir / f"seed_{seed}" / "reference_evaluation.json" for seed in args_.seeds]
        if any(path.is_file() for path in nrec_paths):
            if not all(path.is_file() for path in nrec_paths):
                raise ValueError(f"NRE-C full-data baseline is incomplete for budget {budget}.")
            for seed, path in zip(args_.seeds, nrec_paths, strict=True):
                data = json.loads(path.read_text())
                if data.get("posterior_target") != "theta_given_x_full" or data.get("method") != "nre_c_full_data" or int(data.get("seed", -1)) != seed:
                    raise ValueError(f"NRE-C full-data target/provenance mismatch: {path}")
                ref_rows = data.get("reference_metrics", [])
                validate_ids([r.get("reference_index", -1) for r in ref_rows], args_.reference_ids, str(path))
                samples = path.with_name("reference_posterior_samples.h5")
                _validate_reference_file(full_reference_path, samples, args_.reference_ids, ("theta_true", "x_full"))
                for metric in METRICS:
                    value, n_valid = summarize_observations(ref_rows, metric)
                    field = METRICS[metric][0]
                    stored = data.get("reference_metric_aggregate", {}).get(field)
                    if stored is None or not np.isclose(value, float(stored), rtol=0, atol=1e-12):
                        raise ValueError(f"NRE-C full-data aggregate mismatch: {path}/{metric}")
                    rows.append({"problem":args_.problem,"mechanism":"none","budget":budget,"method":"nre_c_full_data","fraction":0.,"seed":seed,"metric":metric,"target":"full","value":value,"configuration_complete":True,"n_valid_reference_observations":n_valid,"reference_ids":json.dumps(args_.reference_ids),"source":str(path),"backbone":"NRE-C"})
    return pd.DataFrame(rows, columns=SEED_COLUMNS)


def aggregate_long(seed_frame: pd.DataFrame, *, source_map: dict[tuple[str, str, str, str, str], str] | None = None) -> pd.DataFrame:
    keys = ["problem", "mechanism", "fraction", "budget", "method", "backbone", "target", "metric"]
    rows = []
    for key, group in seed_frame.groupby(keys, sort=False, dropna=False):
        if not group.configuration_complete.all():
            raise ValueError(f"Incomplete seed configuration in aggregate group: {key}")
        summary = aggregate_seed_values(group.sort_values("seed").value.astype(float).tolist())
        row = dict(zip(keys, key, strict=True))
        row.update(summary)
        row["source_aggregate"] = "seed_level_metrics"
        rows.append(row)
    return pd.DataFrame(rows, columns=AGG_COLUMNS)


def main() -> None:
    a = args()
    a.outputs_root = a.outputs_root.resolve()
    a.npe_analysis_dir = (a.npe_analysis_dir or Path("outputs/analysis/partial_reference_comparison") / f"{a.problem}_{a.mechanism}").resolve()
    a.analysis_dir = (a.analysis_dir or Path("outputs/analysis/nre_c_comparison") / f"{a.problem}_{a.mechanism}_{fraction_tag(a.fraction)}").resolve()
    npe_seed, npe_summary, analysis_path, npe_metric_settings = _npe_analysis_inputs(a)
    first_source = resolve_path(a.outputs_root, npe_seed.iloc[0].source)
    first_comp = json.loads(first_source.read_text())
    ref_value = first_comp["provenance"]["partial_reference_path"]
    partial_ref = resolve_path(a.outputs_root, ref_value)
    # Verify all NPE partial runs use the same reference artifact before using it for NRE-C.
    for source in npe_seed.source.unique():
        comp = json.loads(resolve_path(a.outputs_root, source).read_text())
        if resolve_path(a.outputs_root, comp["provenance"]["partial_reference_path"]) != partial_ref:
            raise ValueError("NPE partial configurations do not share a single reference artifact.")
    full_csv = a.outputs_root / "outputs/analysis/full_experiment_results_with_diagnostics.csv"
    fullref_value = json.loads((a.outputs_root / "outputs_low_sim_budget/npe_full_data" / f"{a.problem}_full_data/all_results.json").read_text())[0]["reference_path"]
    full_ref = resolve_path(a.outputs_root, fullref_value)
    nrec_seed = _load_partial_nre_c(a, partial_ref, npe_metric_settings)
    baseline_seed = _load_full_data_baselines(a, full_ref, full_csv)
    # Reuse the already validated NPE aggregate means/SDs; the check above confirms
    # they reproduce exactly from the existing NPE seed-level CSV.
    npe_seed = npe_seed.copy()
    npe_seed["backbone"] = "NPE"
    npe_seed["reference_ids"] = json.dumps(a.reference_ids)
    npe_seed["source"] = npe_seed["source"].astype(str)
    npe_combined = pd.concat([npe_seed[SEED_COLUMNS], nrec_seed, baseline_seed], ignore_index=True)
    nrec_aggregate = aggregate_long(nrec_seed)
    npe_groups = []
    for _, row in npe_seed.groupby(["problem", "mechanism", "fraction", "budget", "method", "backbone", "target", "metric"], sort=False).first().reset_index().iterrows():
        summary = npe_summary[(npe_summary.method == row.method) & (npe_summary.budget == row.budget) & (npe_summary.metric == row.metric) & np.isclose(npe_summary.fraction, a.fraction)].iloc[0]
        n = int(summary.n_seeds_partial); mean = float(summary.mean_partial); sd = float(summary.std_partial); half = 1.96 * sd / math.sqrt(n)
        npe_groups.append({**{k:row[k] for k in ("problem","mechanism","fraction","budget","method","backbone","target","metric")},"mean":mean,"std":sd,"ci95_low":mean-half,"ci95_high":mean+half,"n_seeds":n,"source_aggregate":str(analysis_path / "full_vs_partial_summary.csv")})
    # Full-data baselines are smaller legacy files, aggregated directly from their
    # saved per-reference rows and cross-checked against the original NPE CSV.
    baseline_aggregate = aggregate_long(baseline_seed) if not baseline_seed.empty else pd.DataFrame(columns=AGG_COLUMNS)
    comparison_aggregate = pd.concat([pd.DataFrame(npe_groups, columns=AGG_COLUMNS), nrec_aggregate, baseline_aggregate], ignore_index=True)
    nrec_path = full_csv.parent
    a.analysis_dir.mkdir(parents=True, exist_ok=True)
    nrec_seed.to_csv(a.analysis_dir / "nre_c_seed_level_metrics.csv", index=False)
    nrec_aggregate.to_csv(a.analysis_dir / "nre_c_aggregate_metrics.csv", index=False)
    npe_seed[SEED_COLUMNS].to_csv(a.analysis_dir / "npe_seed_level_metrics.csv", index=False)
    pd.DataFrame(npe_groups, columns=AGG_COLUMNS).to_csv(a.analysis_dir / "npe_aggregate_metrics.csv", index=False)
    npe_combined.to_csv(a.analysis_dir / "comparison_seed_level_metrics.csv", index=False)
    comparison_aggregate.to_csv(a.analysis_dir / "comparison_aggregate_metrics.csv", index=False)
    meta = {
        "schema": "gapsbi_npe_nre_c_comparison_v1", "problem": a.problem, "mechanism": a.mechanism,
        "fraction": a.fraction, "budgets": a.budgets, "seeds": a.seeds, "reference_ids": a.reference_ids,
        "partial_reference_path": str(partial_ref), "partial_reference_sha256": hashlib.sha256(partial_ref.read_bytes()).hexdigest(),
        "full_reference_path": str(full_ref), "full_reference_sha256": hashlib.sha256(full_ref.read_bytes()).hexdigest(),
        "partial_target": "theta_given_x_obs_mask", "full_data_target": "theta_given_x_full",
        "within_seed_aggregation": {name: reducer for name, (_field, reducer) in METRICS.items()},
        "across_seed_aggregation": {"mean": "arithmetic mean", "std": "sample standard deviation (ddof=1)", "ci95": "mean +/- 1.96 * std / sqrt(n)"},
        "npe_aggregate_source": str(analysis_path / "full_vs_partial_summary.csv"),
        "npe_aggregate_verified_against_seed_csv": True,
        "reference_metric_settings": npe_metric_settings,
        "reference_metrics_recomputed": False,
        "full_data_nre_c_baseline_included": bool((baseline_seed.backbone == "NRE-C").any()) if not baseline_seed.empty else False,
        "baseline_comparison_note": "Full-data baseline panels use full-data references; missing-data method panels use the shared partial-data reference target. Interpret the baseline column separately.",
    }
    try:
        import subprocess
        meta["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=a.outputs_root, text=True).strip()
    except Exception:
        meta["source_commit"] = None
    try:
        from importlib.metadata import version
        meta["dependency_versions"] = {name: version(name) for name in ("numpy", "pandas", "matplotlib", "torch", "sbi")}
    except Exception:
        meta["dependency_versions"] = {}
    (a.analysis_dir / "provenance.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"output_dir":str(a.analysis_dir),"nre_c_seed_rows":len(nrec_seed),"nre_c_aggregate_rows":len(nrec_aggregate),"npe_seed_rows":len(npe_seed),"comparison_aggregate_rows":len(comparison_aggregate)},sort_keys=True))


if __name__ == "__main__":
    main()
