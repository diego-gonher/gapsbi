#!/usr/bin/env python
"""Run experimental standard L-C2ST against partial-reference observations."""
from __future__ import annotations
import argparse
from pathlib import Path
from gapsbi.evaluation.standard_lc2st import run_from_artifacts, write_standard_lc2st_result

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "calibration", "partial-reference", "output"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--repository-root", default=".")
    p.add_argument("--classifier-seed", type=int, default=101)
    p.add_argument("--calibration-posterior-seed", type=int, default=71001)
    p.add_argument("--reference-posterior-seed", type=int, default=71002)
    p.add_argument("--num-reference-posterior-samples", type=int, default=1000)
    p.add_argument("--num-permutations", type=int, default=199)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--classifier-max-iter", type=int, default=500)
    p.add_argument("--classifier-n-iter-no-change", type=int, default=50)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    result = run_from_artifacts(checkpoint_path=a.checkpoint, calibration_path=a.calibration, partial_reference_path=a.partial_reference, repository_root=Path(a.repository_root), classifier_seed=a.classifier_seed, calibration_posterior_seed=a.calibration_posterior_seed, reference_posterior_seed=a.reference_posterior_seed, num_reference_posterior_samples=a.num_reference_posterior_samples, num_permutations=a.num_permutations, alpha=a.alpha, classifier_max_iter=a.classifier_max_iter, classifier_n_iter_no_change=a.classifier_n_iter_no_change)
    write_standard_lc2st_result(a.output, result, overwrite=a.overwrite)
if __name__ == "__main__": main()
