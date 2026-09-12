from __future__ import annotations

import argparse
from pathlib import Path

from gapsbi.references import (
    plot_glm_reference_predictives,
    plot_glu_reference_predictives,
    plot_lotka_volterra_reference_predictives,
    plot_oup_reference_predictives,
    plot_reference_mcmc_traces,
    plot_reference_posterior_marginals,
    plot_reference_posterior_pairs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot marginal diagnostics for GapSBI reference posterior artifacts."
    )
    parser.add_argument("reference_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--skip-pairs", action="store_true")
    parser.add_argument("--skip-traces", action="store_true")
    args = parser.parse_args()

    paths = plot_reference_posterior_marginals(
        args.reference_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        seed=args.seed,
        bins=args.bins,
    )
    if not args.skip_pairs:
        paths.extend(
            plot_reference_posterior_pairs(
                args.reference_path,
                output_dir=args.output_dir,
                max_samples=args.max_samples,
                seed=args.seed,
                bins=max(args.bins, 80),
            )
        )
    if not args.skip_traces:
        paths.extend(
            plot_reference_mcmc_traces(
                args.reference_path,
                output_dir=args.output_dir,
            )
        )
    paths.extend(
        plot_lotka_volterra_reference_predictives(
            args.reference_path,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            seed=args.seed,
        )
    )
    paths.extend(
        plot_oup_reference_predictives(
            args.reference_path,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            seed=args.seed,
        )
    )
    paths.extend(
        plot_glm_reference_predictives(
            args.reference_path,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            seed=args.seed,
        )
    )
    paths.extend(
        plot_glu_reference_predictives(
            args.reference_path,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            seed=args.seed,
        )
    )
    print(f"Saved {len(paths)} plots")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
