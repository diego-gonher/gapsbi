"""Generate one validated OUP/MNAR partial-data reference artifact."""
from __future__ import annotations
import argparse
from pathlib import Path
from gapsbi.references import (
    DEFAULT_PARTIAL_DATA_REFERENCE_VERSION,
    default_full_reference_path,
    default_partial_reference_path,
    generate_oup_mnar_partial_data_reference_posteriors,
    save_partial_data_reference_posteriors_hdf5,
)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--missing-fraction', type=float, required=True)
    parser.add_argument('--full-reference', type=Path, default=default_full_reference_path('oup'))
    parser.add_argument('--reference-mask-seed', type=int, default=987654)
    parser.add_argument('--posterior-seed', type=int, default=23456)
    parser.add_argument('--num-proposals', type=int, default=1_000_000)
    parser.add_argument('--num-reference-samples', type=int, default=10_000)
    parser.add_argument('--grid-resolution', type=int, default=4800)
    parser.add_argument('--reference-version', default=DEFAULT_PARTIAL_DATA_REFERENCE_VERSION)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    observations, metadata, reference_data = generate_oup_mnar_partial_data_reference_posteriors(
        full_reference_path=args.full_reference,
        missing_fraction=args.missing_fraction,
        reference_mask_seed=args.reference_mask_seed,
        posterior_seed=args.posterior_seed,
        num_proposals=args.num_proposals,
        num_reference_samples=args.num_reference_samples,
        grid_resolution=args.grid_resolution,
    )
    output = args.output or default_partial_reference_path(
        'oup', 'mnar', args.missing_fraction, reference_version=args.reference_version,
    )
    save_partial_data_reference_posteriors_hdf5(
        output, observations=observations, theta_samples=reference_data.pop('theta_samples'),
        metadata=metadata, diagnostics=reference_data, reference_version=args.reference_version,
        overwrite=args.overwrite,
    )
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
