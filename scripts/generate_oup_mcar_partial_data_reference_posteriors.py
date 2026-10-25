"""Generate one OUP/MCAR partial-data reference artifact from the OUP grid posterior."""
from __future__ import annotations

import argparse
from pathlib import Path

from gapsbi.references import (
    DEFAULT_PARTIAL_DATA_REFERENCE_VERSION,
    generate_oup_mcar_partial_data_reference_posteriors,
    save_partial_data_reference_posteriors_hdf5,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--missing-fraction", type=float, required=True)
    parser.add_argument("--full-reference", type=Path, default=Path("references/reference_posteriors_v1/oup_references.h5"))
    parser.add_argument("--reference-mask-seed", type=int, default=987_654)
    parser.add_argument("--posterior-seed", type=int, default=23_456)
    parser.add_argument("--num-reference-samples", type=int, default=10_000)
    parser.add_argument("--grid-resolution", type=int, default=4_800)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--validation-grid-resolution", type=int, default=2_400)
    parser.add_argument("--validation-observations", type=int, nargs="*", default=None)
    parser.add_argument("--reference-version", default=DEFAULT_PARTIAL_DATA_REFERENCE_VERSION)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    observations, metadata, reference_data = generate_oup_mcar_partial_data_reference_posteriors(
        full_reference_path=args.full_reference, missing_fraction=args.missing_fraction,
        reference_mask_seed=args.reference_mask_seed, posterior_seed=args.posterior_seed,
        num_reference_samples=args.num_reference_samples, grid_resolution=args.grid_resolution,
        chunk_size=args.chunk_size, validation_grid_resolution=args.validation_grid_resolution,
        validation_observation_indices=(None if args.validation_observations is None else tuple(args.validation_observations)),
    )
    output = args.output or (
        Path("references") / args.reference_version
        / f"oup_mcar_eps{int(round(args.missing_fraction * 100)):03d}_references.h5"
    )
    theta_samples = reference_data.pop("theta_samples")
    save_partial_data_reference_posteriors_hdf5(
        output, observations=observations, theta_samples=theta_samples, metadata=metadata,
        diagnostics=reference_data, reference_version=args.reference_version, overwrite=args.overwrite,
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
