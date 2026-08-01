from __future__ import annotations

import argparse
from pathlib import Path

from gapsbi.references import (
    DEFAULT_REFERENCE_VERSION,
    generate_glm_reference_posteriors,
    generate_glu_reference_posteriors,
    generate_lotka_volterra_reference_posteriors,
    generate_oup_reference_posteriors,
    save_reference_posteriors_hdf5,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GAPSBI reference posterior HDF5 artifacts."
    )
    parser.add_argument("--problem", choices=["glu", "oup", "glm", "lotka_volterra"], default="glu")
    parser.add_argument("--num-observations", type=int, default=10)
    parser.add_argument("--num-reference-samples", type=int, default=10_000)
    parser.add_argument("--observation-seed", type=int, default=12_345)
    parser.add_argument("--posterior-seed", type=int, default=23_456)
    parser.add_argument("--reference-version", default=DEFAULT_REFERENCE_VERSION)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--prior-bound", type=float, default=None)
    parser.add_argument("--duration", type=int, default=100)
    parser.add_argument("--num-timepoints", type=int, default=50)
    parser.add_argument("--days", type=float, default=20.0)
    parser.add_argument("--observation-noise-scale", type=float, default=0.1)
    parser.add_argument("--stimulus-seed", type=int, default=42)
    parser.add_argument("--simulator-scale", type=float, default=0.1)
    parser.add_argument("--grid-resolution", type=int, default=800)
    parser.add_argument(
        "--validation-grid-resolution",
        dest="validation_grid_resolutions",
        type=int,
        action="append",
        default=None,
        help="Grid resolution to compare against the final OUP grid; may be repeated.",
    )
    parser.add_argument("--spotcheck-grid-resolution", type=int, default=1200)
    parser.add_argument(
        "--spotcheck-observation-index",
        dest="spotcheck_observation_indices",
        type=int,
        action="append",
        default=None,
        help="OUP observation index for high-resolution spot checks; may be repeated.",
    )
    parser.add_argument("--skip-spotcheck", action="store_true")
    parser.add_argument("--num-walkers", type=int, default=128)
    parser.add_argument("--burn-in-steps", type=int, default=3000)
    parser.add_argument("--production-steps", type=int, default=3000)
    parser.add_argument("--initial-scale", type=float, default=0.05)
    parser.add_argument("--trace-num-steps", type=int, default=1000)
    parser.add_argument("--trace-num-walkers", type=int, default=8)
    parser.add_argument("--validation-num-ensembles", type=int, default=2)
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument(
        "--validation-observation-index",
        dest="validation_observation_indices",
        type=int,
        action="append",
        default=None,
        help="GLM observation index for independent ensemble validation; may be repeated.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.problem == "glu":
        observations, metadata, reference_data = generate_glu_reference_posteriors(
            num_observations=args.num_observations,
            num_reference_samples=args.num_reference_samples,
            observation_seed=args.observation_seed,
            posterior_seed=args.posterior_seed,
            dim=args.dim,
            prior_bound=1.0 if args.prior_bound is None else args.prior_bound,
            simulator_scale=args.simulator_scale,
        )
    elif args.problem == "oup":
        validation_grid_resolutions = (
            tuple(args.validation_grid_resolutions)
            if args.validation_grid_resolutions is not None
            else (400,)
        )
        spotcheck_observation_indices = (
            tuple(args.spotcheck_observation_indices)
            if args.spotcheck_observation_indices is not None
            else (0, 1)
        )
        observations, metadata, reference_data = generate_oup_reference_posteriors(
            num_observations=args.num_observations,
            num_reference_samples=args.num_reference_samples,
            observation_seed=args.observation_seed,
            posterior_seed=args.posterior_seed,
            grid_resolution=args.grid_resolution,
            validation_grid_resolutions=validation_grid_resolutions,
            spotcheck_grid_resolution=(
                None if args.skip_spotcheck else args.spotcheck_grid_resolution
            ),
            spotcheck_observation_indices=spotcheck_observation_indices,
        )
    elif args.problem == "glm":
        validation_observation_indices = (
            tuple(args.validation_observation_indices)
            if args.validation_observation_indices is not None
            else (0, 1)
        )
        observations, metadata, reference_data = generate_glm_reference_posteriors(
            num_observations=args.num_observations,
            num_reference_samples=args.num_reference_samples,
            observation_seed=args.observation_seed,
            posterior_seed=args.posterior_seed,
            dim=args.dim,
            prior_bound=2.0 if args.prior_bound is None else args.prior_bound,
            duration=args.duration,
            stimulus_seed=args.stimulus_seed,
            num_walkers=args.num_walkers,
            burn_in_steps=args.burn_in_steps,
            production_steps=args.production_steps,
            initial_scale=args.initial_scale,
            trace_num_steps=args.trace_num_steps,
            trace_num_walkers=args.trace_num_walkers,
            validation_num_ensembles=args.validation_num_ensembles,
            validation_observation_indices=validation_observation_indices,
        )
    elif args.problem == "lotka_volterra":
        validation_observation_indices = (
            tuple(args.validation_observation_indices)
            if args.validation_observation_indices is not None
            else (0, 1)
        )
        observations, metadata, reference_data = generate_lotka_volterra_reference_posteriors(
            num_observations=args.num_observations,
            num_reference_samples=args.num_reference_samples,
            observation_seed=args.observation_seed,
            posterior_seed=args.posterior_seed,
            num_timepoints=args.num_timepoints,
            days=args.days,
            observation_noise_scale=args.observation_noise_scale,
            num_walkers=args.num_walkers,
            burn_in_steps=args.burn_in_steps,
            production_steps=args.production_steps,
            initial_scale=args.initial_scale,
            trace_num_steps=args.trace_num_steps,
            trace_num_walkers=args.trace_num_walkers,
            validation_num_ensembles=args.validation_num_ensembles,
            validation_observation_indices=validation_observation_indices,
            progress=args.progress,
        )
    else:
        raise ValueError(f"Unsupported problem: {args.problem}")

    output_path = args.output
    if output_path is None:
        output_path = (
            Path("references")
            / args.reference_version
            / f"{args.problem}_references.h5"
        )
    theta_samples = reference_data.pop("theta_samples")
    save_reference_posteriors_hdf5(
        output_path,
        observations=observations,
        theta_samples=theta_samples,
        metadata=metadata,
        diagnostics=reference_data,
        reference_version=args.reference_version,
        overwrite=args.overwrite,
    )
    print(f"Saved: {output_path}")
    print(
        "reference_posterior/theta_samples="
        f"{theta_samples.shape}, observations/x_full={observations['x_full'].shape}"
    )


if __name__ == "__main__":
    main()
