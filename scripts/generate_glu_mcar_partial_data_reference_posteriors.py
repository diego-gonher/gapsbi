from __future__ import annotations

import argparse
from pathlib import Path

from gapsbi.references import (
    DEFAULT_PARTIAL_DATA_REFERENCE_VERSION,
    generate_glu_mcar_partial_data_reference_posteriors,
    save_glu_mcar_partial_data_reference_posteriors_hdf5,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GLU MCAR partial-data reference posterior artifacts."
    )
    parser.add_argument("--num-observations", type=int, default=10)
    parser.add_argument("--num-reference-samples", type=int, default=10_000)
    parser.add_argument("--observation-seed", type=int, default=12_345)
    parser.add_argument("--posterior-seed", type=int, default=23_456)
    parser.add_argument("--reference-mask-seed", type=int, default=987_654)
    parser.add_argument("--missing-fraction", type=float, required=True)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--prior-bound", type=float, default=1.0)
    parser.add_argument("--simulator-scale", type=float, default=0.1)
    parser.add_argument("--reference-version", default=DEFAULT_PARTIAL_DATA_REFERENCE_VERSION)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    observations, metadata, reference_data = (
        generate_glu_mcar_partial_data_reference_posteriors(
            num_observations=args.num_observations,
            num_reference_samples=args.num_reference_samples,
            observation_seed=args.observation_seed,
            posterior_seed=args.posterior_seed,
            reference_mask_seed=args.reference_mask_seed,
            missing_fraction=args.missing_fraction,
            dim=args.dim,
            prior_bound=args.prior_bound,
            simulator_scale=args.simulator_scale,
        )
    )
    output_path = args.output or (
        Path("references")
        / args.reference_version
        / f"glu_mcar_eps{int(round(args.missing_fraction * 100)):03d}_references.h5"
    )
    theta_samples = reference_data.pop("theta_samples")
    save_glu_mcar_partial_data_reference_posteriors_hdf5(
        output_path,
        observations=observations,
        theta_samples=theta_samples,
        metadata=metadata,
        diagnostics=reference_data,
        reference_version=args.reference_version,
        overwrite=args.overwrite,
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
