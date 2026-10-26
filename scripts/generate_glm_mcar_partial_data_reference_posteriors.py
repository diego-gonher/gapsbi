"""Generate one GLM/MCAR partial-data reference artifact.

The script is intentionally opt-in per missing fraction.  It consumes the
immutable full-data GLM reference observations and never overwrites an artifact
unless ``--overwrite`` is explicitly supplied.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from gapsbi.references import (
    DEFAULT_PARTIAL_DATA_REFERENCE_VERSION,
    default_full_reference_path,
    default_partial_reference_path,
    generate_glm_mcar_partial_data_reference_posteriors,
    save_partial_data_reference_posteriors_hdf5,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--missing-fraction", type=float, required=True)
    parser.add_argument("--full-reference", type=Path, default=default_full_reference_path("glm"))
    parser.add_argument("--reference-mask-seed", type=int, default=987_654)
    parser.add_argument("--posterior-seed", type=int, default=23_456)
    parser.add_argument("--num-reference-samples", type=int, default=10_000)
    parser.add_argument("--num-mh-chains", type=int, default=4)
    parser.add_argument("--mh-steps", type=int, default=50_000)
    parser.add_argument("--mh-burn-in-steps", type=int, default=2_000)
    parser.add_argument("--mh-thin", type=int, default=2)
    parser.add_argument("--max-rank-rhat", type=float, default=1.01)
    parser.add_argument("--min-bulk-ess", type=float, default=1_000.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-step-multiplier", type=float, default=2.0)
    parser.add_argument("--reference-version", default=DEFAULT_PARTIAL_DATA_REFERENCE_VERSION)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    observations, metadata, reference_data = generate_glm_mcar_partial_data_reference_posteriors(
        full_reference_path=args.full_reference, missing_fraction=args.missing_fraction,
        reference_mask_seed=args.reference_mask_seed, posterior_seed=args.posterior_seed,
        num_reference_samples=args.num_reference_samples, num_mh_chains=args.num_mh_chains,
        mh_steps=args.mh_steps, mh_burn_in_steps=args.mh_burn_in_steps, mh_thin=args.mh_thin,
        max_rank_rhat=args.max_rank_rhat, min_bulk_ess=args.min_bulk_ess,
        max_attempts=args.max_attempts, retry_step_multiplier=args.retry_step_multiplier,
    )
    output = args.output or default_partial_reference_path(
        "glm", "mcar", args.missing_fraction,
        reference_version=args.reference_version,
    )
    theta_samples = reference_data.pop("theta_samples")
    save_partial_data_reference_posteriors_hdf5(
        output, observations=observations, theta_samples=theta_samples, metadata=metadata,
        diagnostics=reference_data, reference_version=args.reference_version, overwrite=args.overwrite,
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
