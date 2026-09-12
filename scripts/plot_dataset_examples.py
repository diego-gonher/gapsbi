from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from gapsbi.diagnostics.plots import (
    plot_dataset_examples,
    plot_lotka_volterra_dataset_examples,
    plot_spatial_sir_dataset_examples,
    plot_vector_dataset_examples,
)
from gapsbi.io import load_gapsbi_hdf5


def resolve_plot_type(plot_type: str, metadata: dict) -> str:
    """Resolve CLI plot type, using metadata for auto mode."""
    if plot_type != "auto":
        return plot_type

    task = metadata.get("task")
    simulator_name = metadata.get("simulator", {}).get("name")
    if task == "spatial_sir" or simulator_name == "spatial_sir":
        return "spatial_sir"
    if task == "lotka_volterra" or simulator_name == "lotka_volterra":
        return "lotka_volterra"
    if task in {"glu", "glm"}:
        return "vector"
    return "timeseries"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot examples from a GapSBI dataset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--num-examples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default="outputs_local/data_generation_examples/dataset_examples.png",
    )
    parser.add_argument("--log-y", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--plot-type",
        choices=["timeseries", "vector", "spatial_sir", "lotka_volterra", "auto"],
        default="auto",
    )
    args = parser.parse_args()

    if args.num_examples < 1:
        raise ValueError("--num-examples must be at least 1.")

    dataset, metadata = load_gapsbi_hdf5(args.input)
    if args.split not in dataset:
        raise ValueError(f"Split {args.split!r} not found in dataset.")

    split = dataset[args.split]
    n_examples = split["x_full"].shape[0]
    if n_examples == 0:
        raise ValueError(f"Split {args.split!r} contains no examples.")

    rng = np.random.default_rng(args.seed)
    count = min(args.num_examples, n_examples)
    indices = rng.choice(n_examples, size=count, replace=False)

    output_path = Path(args.output)
    resolved_plot_type = resolve_plot_type(args.plot_type, metadata)
    if resolved_plot_type == "spatial_sir":
        simulator_metadata = metadata.get("simulator", {})
        original_x_shape = tuple(simulator_metadata["original_x_shape"])
        plot_spatial_sir_dataset_examples(
            split,
            indices=indices,
            output_path=output_path,
            original_x_shape=original_x_shape,
            split_name=args.split,
        )
    elif resolved_plot_type == "vector":
        if args.log_y:
            print("Warning: --log-y is ignored for vector plots.")
        plot_vector_dataset_examples(
            split,
            indices=indices,
            output_path=output_path,
            split_name=args.split,
        )
    elif resolved_plot_type == "lotka_volterra":
        simulator_metadata = metadata.get("simulator", {})
        timepoints = simulator_metadata.get("timepoints")
        plot_lotka_volterra_dataset_examples(
            split,
            indices=indices,
            output_path=output_path,
            split_name=args.split,
            timepoints=None if timepoints is None else np.asarray(timepoints, dtype=float),
            log_y=True,
        )
    else:
        plot_dataset_examples(
            split,
            indices=indices,
            output_path=output_path,
            split_name=args.split,
            log_y=args.log_y,
        )

    print(f"Input: {args.input}")
    print(f"Split: {args.split}")
    print(f"Plot type: {resolved_plot_type}")
    print(f"Selected indices: {indices.tolist()}")
    print(f"Output: {output_path}")

    if args.show:
        image = mpimg.imread(output_path)
        plt.figure(figsize=(10, 6))
        plt.imshow(image)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
