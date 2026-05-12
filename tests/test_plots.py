import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

from gapsbi.diagnostics.plots import format_theta, plot_vector_dataset_examples, plot_vector_example


def test_format_theta_returns_empty_string_for_none() -> None:
    assert format_theta(None) == ""


def test_format_theta_shows_all_values_for_short_theta() -> None:
    assert format_theta(np.array([2.105, 0.09])) == "theta=[2.105, 0.09]"


def test_format_theta_truncates_long_theta() -> None:
    theta = np.array([0.792, 0.667, -0.064, 1.0])

    assert format_theta(theta) == "theta_dim=4, theta[:3]=[0.792, 0.667, -0.064], ..."


def test_plot_vector_example_runs_with_small_arrays() -> None:
    fig, ax = plt.subplots()

    plot_vector_example(
        ax,
        x_full=np.array([1.0, 2.0, 3.0]),
        x_obs=np.array([1.0, 0.0, 3.0]),
        mask=np.array([1, 0, 1]),
        theta=np.array([0.1, 0.2]),
    )

    assert ax.get_xlabel() == "feature"
    assert ax.get_ylabel() == "x"
    plt.close(fig)


def test_plot_vector_dataset_examples_saves_file(tmp_path) -> None:
    dataset_split = {
        "theta": np.ones((2, 3)),
        "x_full": np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]),
        "x_obs": np.array([[1.0, 0.0, 3.0], [0.0, 2.0, 1.0]]),
        "mask": np.array([[1, 0, 1], [0, 1, 1]]),
    }
    output_path = tmp_path / "vector_examples.png"

    plot_vector_dataset_examples(dataset_split, np.array([0, 1]), output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_script_auto_mode_chooses_vector_for_glu() -> None:
    script_path = Path(__file__).parents[1] / "scripts" / "plot_dataset_examples.py"
    spec = importlib.util.spec_from_file_location("plot_dataset_examples_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.resolve_plot_type("auto", {"task": "glu"}) == "vector"
    assert module.resolve_plot_type("auto", {"task": "ricker"}) == "timeseries"
    assert module.resolve_plot_type("timeseries", {"task": "glu"}) == "timeseries"
