from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gapsbi.datasets import generate_dataset
from gapsbi.io import save_gapsbi_hdf5
from gapsbi.masks import (
    BlockMCARMask,
    CoordinateMARMask,
    LotkaVolterraLogTotalMNARMask,
    LotkaVolterraTimeBlockMCARMask,
    LotkaVolterraTimeMARMask,
    PointMCARMask,
    SelfCensoringMNARMask,
)
from gapsbi.simulators import (
    GLMSimulator,
    GLUSimulator,
    HodgkinHuxleySimulator,
    LotkaVolterraSimulator,
    OUPSimulator,
    RickerSimulator,
    SpatialSIRSimulator,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a GAPSBI HDF5 dataset.")
    parser.add_argument(
        "--task",
        choices=[
            "ricker",
            "oup",
            "glu",
            "glm",
            "spatial_sir",
            "hodgkin_huxley",
            "lotka_volterra",
        ],
        default="ricker",
    )
    parser.add_argument(
        "--mask",
        choices=[
            "point_mcar",
            "block_mcar",
            "self_censoring_mnar",
            "coordinate_mar",
            "lv_time_block_mcar",
            "lv_time_mar",
            "lv_log_total_mnar",
        ],
        default="point_mcar",
    )
    parser.add_argument("--missing-fraction", type=float, default=0.25)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument(
        "--mnar-score-transform",
        choices=["identity", "log1p", "abs"],
        default="identity",
    )
    parser.add_argument("--mar-mode", choices=["increasing", "decreasing", "middle"], default="increasing")
    parser.add_argument("--mar-floor", type=float, default=0.05)
    parser.add_argument("--mar-max-probability", type=float, default=0.95)
    parser.add_argument("--mar-middle-width", type=float, default=0.2)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--prior-bound", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--summary", choices=["sufficient", "raw"], default="sufficient")
    parser.add_argument("--simulator-scale", type=float, default=0.1)
    parser.add_argument("--num-timepoints", type=int, default=50)
    parser.add_argument("--days", type=float, default=20.0)
    parser.add_argument("--observation-noise-scale", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--t-on", type=float, default=10.0)
    parser.add_argument("--curr-level", type=float, default=5e-4)
    parser.add_argument("--downsample", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--measurement-time", type=float, default=0.25)
    parser.add_argument("--simulation-step-size", type=float, default=0.01)
    parser.add_argument("--initial-infection-rate", type=float, default=3.0)
    parser.add_argument("--n-train", type=int, default=1_000)
    parser.add_argument("--n-val", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    args = parser.parse_args()

    if args.task == "ricker":
        simulator = RickerSimulator()
    elif args.task == "oup":
        simulator = OUPSimulator()
    elif args.task == "glu":
        simulator = GLUSimulator(dim=args.dim, simulator_scale=args.simulator_scale)
    elif args.task == "glm":
        simulator = GLMSimulator(
            dim=args.dim,
            prior_bound=args.prior_bound,
            duration=100 if args.duration is None else args.duration,
            summary=args.summary,
        )
    elif args.task == "spatial_sir":
        simulator = SpatialSIRSimulator(
            lattice_shape=(args.grid_size, args.grid_size),
            measurement_time=args.measurement_time,
            simulation_step_size=args.simulation_step_size,
            initial_infection_rate=args.initial_infection_rate,
        )
    elif args.task == "hodgkin_huxley":
        simulator = HodgkinHuxleySimulator(
            duration=120.0 if args.duration is None else args.duration,
            dt=args.dt,
            t_on=args.t_on,
            curr_level=args.curr_level,
            downsample=args.downsample,
        )
    else:
        simulator = LotkaVolterraSimulator(
            num_timepoints=args.num_timepoints,
            days=args.days,
            observation_noise_scale=args.observation_noise_scale,
        )
    if args.mask == "point_mcar":
        mask_generator = PointMCARMask(missing_fraction=args.missing_fraction)
    elif args.mask == "block_mcar":
        mask_generator = BlockMCARMask(
            missing_fraction=args.missing_fraction,
            block_size=args.block_size,
        )
    elif args.mask == "self_censoring_mnar":
        mask_generator = SelfCensoringMNARMask(
            missing_fraction=args.missing_fraction,
            score_transform=args.mnar_score_transform,
        )
    elif args.mask == "coordinate_mar":
        mask_generator = CoordinateMARMask(
            missing_fraction=args.missing_fraction,
            mode=args.mar_mode,
            floor=args.mar_floor,
            max_probability=args.mar_max_probability,
            middle_width=args.mar_middle_width,
        )
    elif args.mask == "lv_time_block_mcar":
        mask_generator = LotkaVolterraTimeBlockMCARMask(
            missing_fraction=args.missing_fraction,
            block_size=args.block_size,
        )
    elif args.mask == "lv_time_mar":
        mask_generator = LotkaVolterraTimeMARMask(
            missing_fraction=args.missing_fraction,
            mode=args.mar_mode,
            floor=args.mar_floor,
            max_probability=args.mar_max_probability,
            middle_width=args.mar_middle_width,
        )
    else:
        mask_generator = LotkaVolterraLogTotalMNARMask(
            missing_fraction=args.missing_fraction,
        )

    dataset = generate_dataset(
        simulator=simulator,
        mask_generator=mask_generator,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seed=args.seed,
        progress=args.progress,
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
