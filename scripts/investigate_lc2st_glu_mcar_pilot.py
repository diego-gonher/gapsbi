#!/usr/bin/env python3
"""Sensitivity and classifier-health checks for the GLU standard L-C2ST pilot."""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from gapsbi.evaluation.lc2st import mask_augmented_context
from gapsbi.references import sample_glu_mcar_partial_data_reference_posterior


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/canonical_v1/glu/mcar/glu_mcar_eps025_seed123.h5"
RUN = ROOT / "outputs_low_sim_budget/npe_mask_augmentation/glu/mcar/glu_mcar_eps025/seed_101"
PARTIAL_REFERENCE = ROOT / "references/partial_data_references_v1/glu_mcar_eps025_references.h5"
OUT = ROOT / "outputs/diagnostics/lc2st_glu_mcar_eps025_zero_mask_low_seed101/sensitivity_investigation.json"


def encode(value):
    if isinstance(value, (np.integer, np.floating)): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=encode), encoding="utf-8")
    temp.replace(path)


def select_saved_draws(path: Path, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    with h5py.File(path, "r") as h5:
        stored = h5["theta_posterior"]
        selected, indices = [], []
        for i in range(stored.shape[0]):
            row = stored[i]
            valid = np.flatnonzero(np.isfinite(row).all(axis=1))
            if not len(valid): raise ValueError(f"No finite saved posterior draw at calibration index {i}.")
            draw = int(rng.choice(valid)); selected.append(row[draw]); indices.append(draw)
    return np.asarray(selected, dtype=np.float32), np.asarray(indices, dtype=np.int64)


def partial_draws(x_obs: np.ndarray, mask: np.ndarray, n: int, seed: int, *, shift: float = 0.0, scale: float = .1) -> np.ndarray:
    """Exact GLU MCAR draws, optionally with controlled observed-coordinate error."""
    x_shifted = np.asarray(x_obs, dtype=float).copy()
    x_shifted[mask.astype(bool)] += shift
    return sample_glu_mcar_partial_data_reference_posterior(
        x_obs=x_shifted, mask=mask, prior_low=np.full(x_obs.shape[1], -1.), prior_high=np.full(x_obs.shape[1], 1.),
        simulator_scale=scale, num_reference_samples=n, rng=np.random.default_rng(seed),
    ).astype(np.float32)


def probability_summary(probability_class_zero: np.ndarray, labels_class_zero: np.ndarray) -> dict:
    p = np.asarray(probability_class_zero); y = np.asarray(labels_class_zero)
    bins = np.linspace(0., 1., 11)
    bin_ids = np.digitize(p, bins[1:-1])
    ece = sum(np.abs(p[bin_ids == i].mean() - y[bin_ids == i].mean()) * (bin_ids == i).mean()
              for i in range(10) if np.any(bin_ids == i))
    observed, predicted = calibration_curve(y, p, n_bins=10, strategy="uniform")
    auc = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None
    return {"mean": float(p.mean()), "std": float(p.std()), "q01_q05_q50_q95_q99": np.quantile(p, [.01,.05,.5,.95,.99]),
            "fraction_within_0.45_0.55": float(((p >= .45) & (p <= .55)).mean()),
            "accuracy": float(accuracy_score(y, p >= .5)), "auc": auc,
            "brier": float(brier_score_loss(y, p)), "ece_10_bin": float(ece),
            "calibration_predicted": predicted, "calibration_observed": observed}


def train_and_evaluate(theta, context, q_cal, q_ref, x_ref, *, seed: int, null_trials: int, hidden: tuple[int, ...], label: str) -> dict:
    from sbi.diagnostics.lc2st import LC2ST
    test = LC2ST(
        thetas=torch.as_tensor(theta), xs=torch.as_tensor(context), posterior_samples=torch.as_tensor(q_cal),
        seed=seed, num_folds=1, num_ensemble=1, classifier="mlp", z_score=False,
        classifier_kwargs={"hidden_layer_sizes": hidden, "max_iter": 500, "n_iter_no_change": 50},
        num_trials_null=null_trials, permutation=True, device="cpu",
    )
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        test.train_on_observed_data(seed=seed, verbosity=0)
        test.train_under_null_hypothesis(verbosity=0)
    clf = test.trained_clfs[0]
    joint_p = np.concatenate([q_cal, context], axis=1)
    joint_q = np.concatenate([theta, context], axis=1)
    all_x = np.concatenate([joint_p, joint_q]); labels_zero = np.r_[np.ones(len(joint_p)), np.zeros(len(joint_q))]
    probs_zero = clf.predict_proba(all_x)[:, 0]
    rows = []
    for index in range(len(q_ref)):
        theta_o, x_o = torch.as_tensor(q_ref[index]), torch.as_tensor(x_ref[index])
        probs, scores = test.get_scores(theta_o, x_o, test.trained_clfs, return_probs=True)
        null_stats = test.get_statistics_under_null_hypothesis(theta_o, x_o, verbosity=0)
        p = np.asarray(probs).reshape(-1)
        rows.append({"reference_index": index, "statistic": float(np.mean(scores)),
                     "p_value": float((float(np.mean(scores)) < null_stats).mean()),
                     "probability_class_zero": probability_summary(p, np.ones(len(p))),
                     "null_statistic_mean": float(np.mean(null_stats)), "null_statistic_std": float(np.std(null_stats))})
    internal_validation = {"best_validation_score": getattr(clf, "best_validation_score_", None),
                           "final_validation_score": (getattr(clf, "validation_scores_", [None]) or [None])[-1],
                           "n_iter": getattr(clf, "n_iter_", None), "loss": getattr(clf, "loss_", None)}
    return {"label": label, "seed": seed, "hidden_layer_sizes": hidden, "null_trials": null_trials,
            "fit_seconds": time.perf_counter() - started, "joint_classifier": probability_summary(probs_zero, labels_zero),
            "internal_validation": internal_validation, "reference_results": rows}


def main() -> None:
    with h5py.File(DATASET, "r") as h5:
        theta = h5["test/theta"][:].astype(np.float32); x_obs = h5["test/x_obs"][:].astype(np.float32); mask = h5["test/mask"][:].astype(np.int8)
    with h5py.File(RUN / "reference_posterior_samples.h5", "r") as h5:
        q_saved_ref = h5["theta_posterior"][:].astype(np.float32)
        saved_ref_x_obs, saved_ref_mask = h5["x_obs"][:], h5["mask"][:]
        x_ref = mask_augmented_context(saved_ref_x_obs, saved_ref_mask).astype(np.float32)
    with h5py.File(PARTIAL_REFERENCE, "r") as h5:
        ref_obs = h5["observations"]
        if not np.array_equal(ref_obs["x_obs"][:], saved_ref_x_obs) or not np.array_equal(ref_obs["mask"][:], saved_ref_mask):
            # x_ref first half is x_obs, explicitly retain a clear artifact check.
            raise ValueError("Reference x_obs alignment failed.")
        ref_x_obs, ref_mask = ref_obs["x_obs"][:], ref_obs["mask"][:]
    context = mask_augmented_context(x_obs, mask).astype(np.float32)
    q_saved_cal, draw_indices = select_saved_draws(RUN / "posterior_samples.h5", 7101)

    # Correct analytic draws and two controlled alternatives: a 1-sigma location
    # shift on observed coordinates, and a factor-two standard-deviation shrinkage.
    scenarios = {
        "saved_estimator": (q_saved_cal, q_saved_ref),
        "analytic_exact": (partial_draws(x_obs, mask, 1, 8101)[:, 0], partial_draws(ref_x_obs, ref_mask, 10_000, 8102)),
        "mean_shift_plus_0p10": (partial_draws(x_obs, mask, 1, 8201, shift=.10)[:, 0], partial_draws(ref_x_obs, ref_mask, 10_000, 8202, shift=.10)),
        "observed_sd_half": (partial_draws(x_obs, mask, 1, 8301, scale=.05)[:, 0], partial_draws(ref_x_obs, ref_mask, 10_000, 8302, scale=.05)),
    }
    started = time.perf_counter()
    primary = [train_and_evaluate(theta, context, qcal, qref, x_ref, seed=101, null_trials=20, hidden=(100,100), label=name)
               for name, (qcal, qref) in scenarios.items()]
    # Cheap architecture sensitivity: same calibration set and controlled targets,
    # fewer null fits only because it is not used for a formal decision.
    alternatives = []
    for hidden in ((30,), (200, 100)):
        for name in ("analytic_exact", "mean_shift_plus_0p10", "observed_sd_half"):
            qcal, qref = scenarios[name]
            alternatives.append(train_and_evaluate(theta, context, qcal, qref, x_ref, seed=303, null_trials=10, hidden=hidden, label=name))
    existing = json.loads((RUN / "reference_metrics_with_partial_references.json").read_text())
    per_observation_existing = {key: value["rows"] for key, value in (("full_reference", existing["full_reference"]), ("partial_reference", existing["partial_reference"]))}
    atomic_json(OUT, {"schema": "gapsbi_lc2st_sensitivity_v1", "configuration": {"problem":"glu","mechanism":"mcar","fraction":.25,"method":"npe_mask_augmentation","budget":"low","seed":101},
                      "implementation_note": "sbi 0.26.1 LC2ST with z_score=False; its permutation-null path re-applies x scaling when z_score=True.",
                      "calibration": {"n":1000,"context":"concat(x_obs,mask)","saved_draw_indices":draw_indices,"independent_of_reference_observations":True},
                      "controlled_scenarios": {"analytic_exact":"exact MCAR analytic posterior", "mean_shift_plus_0p10":"+0.10 only on observed-coordinate posterior means", "observed_sd_half":"0.05 rather than 0.10 observed-coordinate likelihood scale"},
                      "primary_runs":primary,"architecture_runs":alternatives,"existing_reference_c2st":per_observation_existing,"runtime_seconds":time.perf_counter()-started,
                      "limitations":["Null p-values have resolution 1/20 in primary runs and 1/10 in architecture checks.","Joint-classifier accuracy is an in-sample health diagnostic; internal MLP validation is reported separately."]})
    print(OUT)


if __name__ == "__main__": main()
