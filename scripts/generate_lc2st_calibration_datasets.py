#!/usr/bin/env python3
"""Generate independent MCAR calibration pairs for experimental L-C2ST.

The files use the normal GapSBI HDF5 contract but live outside canonical
benchmark data. They contain test-only prior-predictive pairs and must never
be used to train or select an NPE estimator.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from gapsbi.datasets import generate_dataset
from gapsbi.io import save_gapsbi_hdf5
from gapsbi.masks import LotkaVolterraTimeBlockMCARMask, PointMCARMask
from gapsbi.simulators import GLMSimulator, GLUSimulator, LotkaVolterraSimulator, OUPSimulator


CALIBRATION_ROLE = "lc2st_prior_predictive_calibration"
MISSING_FRACTION = 0.10
TASK_SEEDS = {
    "oup": 31_101,
    "glm": 31_201,
    "glu": 31_301,
    "lotka_volterra": 31_401,
}


def calibration_filename(task: str, seed: int) -> str:
    """Return the stable experimental artifact name for a task and seed."""
    if task == "glm":
        return f"glm_raw_mcar_eps010_seed{seed}.h5"
    if task == "lotka_volterra":
        return f"lotka_volterra_time_block_mcar_eps010_seed{seed}.h5"
    return f"{task}_mcar_eps010_seed{seed}.h5"


def build_task_components(task: str):
    """Construct the canonical simulator and its 10%-MCAR mask generator."""
    if task == "oup":
        return OUPSimulator(), PointMCARMask(MISSING_FRACTION)
    if task == "glm":
        return GLMSimulator(dim=10, duration=100, summary="raw"), PointMCARMask(MISSING_FRACTION)
    if task == "glu":
        return GLUSimulator(dim=10, simulator_scale=0.1), PointMCARMask(MISSING_FRACTION)
    if task == "lotka_volterra":
        return (
            LotkaVolterraSimulator(num_timepoints=50, days=20.0, observation_noise_scale=0.1),
            LotkaVolterraTimeBlockMCARMask(MISSING_FRACTION, block_size=5),
        )
    raise ValueError(f"Unsupported L-C2ST task: {task!r}.")


def generate_calibration_dataset(task: str, num_calibration: int, seed: int):
    """Generate test-only independent calibration pairs using canonical machinery."""
    if num_calibration <= 0:
        raise ValueError("num_calibration must be positive.")
    simulator, mask_generator = build_task_components(task)
    return generate_dataset(
        simulator=simulator,
        mask_generator=mask_generator,
        n_train=0,
        n_val=0,
        n_test=num_calibration,
        seed=seed,
        progress=False,
    ), simulator, mask_generator


def source_commit() -> str:
    """Return the checkout revision so generated data records its exact source."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def write_calibration_dataset(
    *, task: str, num_calibration: int, seed: int, output_root: Path,
    commit: str, overwrite: bool,
) -> Path:
    """Write one experimental calibration HDF5 artifact and return its path."""
    dataset, simulator, mask_generator = generate_calibration_dataset(task, num_calibration, seed)
    output = output_root / task / "mcar" / calibration_filename(task, seed)
    metadata = {
        "task": task,
        "mechanism": "mcar",
        "missing_fraction": MISSING_FRACTION,
        "seed": int(seed),
        "source_commit": commit,
        "calibration_role": CALIBRATION_ROLE,
        "simulator": simulator.metadata(),
        "mask": mask_generator.metadata(),
    }
    save_gapsbi_hdf5(output, dataset, metadata=metadata, overwrite=overwrite)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASK_SEEDS), action="append", help="Repeat to select tasks; default: all.")
    parser.add_argument("--num-calibration", type=int, default=10_000)
    parser.add_argument("--output-root", type=Path, default=Path("data/experimental/lc2st_calibration_v1"))
    parser.add_argument("--source-commit", default=None, help="Override the recorded Git revision (normally detected).")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commit = args.source_commit or source_commit()
    tasks = args.task or list(TASK_SEEDS)
    for task in tasks:
        output = write_calibration_dataset(
            task=task,
            num_calibration=args.num_calibration,
            seed=TASK_SEEDS[task],
            output_root=args.output_root,
            commit=commit,
            overwrite=args.overwrite,
        )
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
