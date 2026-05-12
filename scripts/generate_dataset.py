from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gapsbi.datasets import generate_dataset
from gapsbi.io import save_gapsbi_hdf5
from gapsbi.masks import BlockMCARMask, PointMCARMask
from gapsbi.simulators import GLMSimulator, GLUSimulator, OUPSimulator, RickerSimulator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a GAPSBI HDF5 dataset.")
    parser.add_argument("--task", choices=["ricker", "oup", "glu", "glm"], default="ricker")
    parser.add_argument("--mask", choices=["point_mcar", "block_mcar"], default="point_mcar")
    parser.add_argument("--missing-fraction", type=float, default=0.25)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--prior-bound", type=float, default=2.0)
    parser.add_argument("--duration", type=int, default=100)
    parser.add_argument("--summary", choices=["sufficient", "raw"], default="sufficient")
    parser.add_argument("--simulator-scale", type=float, default=0.1)
    parser.add_argument("--n-train", type=int, default=1_000)
    parser.add_argument("--n-val", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.task == "ricker":
        simulator = RickerSimulator()
    elif args.task == "oup":
        simulator = OUPSimulator()
    elif args.task == "glu":
        simulator = GLUSimulator(dim=args.dim, simulator_scale=args.simulator_scale)
    else:
        simulator = GLMSimulator(
            dim=args.dim,
            prior_bound=args.prior_bound,
            duration=args.duration,
            summary=args.summary,
        )
    if args.mask == "point_mcar":
        mask_generator = PointMCARMask(missing_fraction=args.missing_fraction)
    else:
        mask_generator = BlockMCARMask(
            missing_fraction=args.missing_fraction,
            block_size=args.block_size,
        )

    dataset = generate_dataset(
        simulator=simulator,
        mask_generator=mask_generator,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seed=args.seed,
    )
    metadata = {
        "task": args.task,
        "simulator": simulator.metadata(),
        "mask": mask_generator.metadata(),
        "seed": args.seed,
    }

    output_path = Path(args.output)
    save_gapsbi_hdf5(output_path, dataset, metadata=metadata, overwrite=args.overwrite)

    print(f"Saved: {output_path}")
    for split_name, split_data in dataset.items():
        missing_fraction = 1.0 - float(np.mean(split_data["mask"]))
        print(
            f"{split_name}: "
            f"theta={split_data['theta'].shape}, "
            f"x_full={split_data['x_full'].shape}, "
            f"missing_fraction={missing_fraction:.3f}"
        )


if __name__ == "__main__":
    main()
