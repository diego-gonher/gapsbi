from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def format_theta(theta: np.ndarray | None, max_values: int = 3, precision: int = 3) -> str:
    """Format theta compactly for subplot titles."""
    if theta is None:
        return ""
    if max_values < 1:
        raise ValueError("max_values must be at least 1.")

    theta_array = np.asarray(theta, dtype=float).ravel()
    shown = theta_array[:max_values]
    shown_text = _format_theta_values(shown, precision)

    if theta_array.size <= max_values:
        return f"theta={shown_text}"
    return f"theta_dim={theta_array.size}, theta[:{max_values}]={shown_text}, ..."


def _format_theta_values(values: np.ndarray, precision: int) -> str:
    formatted = []
    for value in values:
        text = f"{float(value):.{precision}f}".rstrip("0").rstrip(".")
        formatted.append("0" if text == "-0" else text)
    return "[" + ", ".join(formatted) + "]"


def plot_timeseries_example(
    ax: Any,
    x_full: np.ndarray,
    x_obs: np.ndarray,
    mask: np.ndarray,
    theta: np.ndarray | None = None,
    log_y: bool = False,
) -> None:
    """Plot one complete, observed, and missing time-series example."""
    x_full = np.asarray(x_full)
    x_obs = np.asarray(x_obs)
    mask = np.asarray(mask).astype(bool)

    if x_full.ndim != 1 or x_obs.shape != x_full.shape or mask.shape != x_full.shape:
        raise ValueError("x_full, x_obs, and mask must be 1D arrays with matching shapes.")

    time = np.arange(x_full.shape[0])
    ax.plot(time, x_full, color="C0", linewidth=1.5, label="x_full")
    ax.scatter(time[mask], x_obs[mask], color="C1", s=18, label="observed", zorder=3)

    missing = ~mask
    if np.any(missing):
        ax.scatter(
            time[missing],
            x_full[missing],
            color="red",
            marker="x",
            s=24,
            label="missing",
            zorder=4,
        )

    theta_title = format_theta(theta)
    if theta_title:
        ax.set_title(theta_title)
    if log_y:
        ax.set_yscale("symlog")

    ax.set_xlabel("time")
    ax.set_ylabel("x")
    ax.legend(fontsize="small")


def plot_dataset_examples(
    dataset_split: dict[str, np.ndarray],
    indices: np.ndarray,
    output_path: str | os.PathLike[str],
    split_name: str = "train",
    log_y: bool = False,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Plot selected time-series examples from one dataset split and save them."""
    x_full = np.asarray(dataset_split["x_full"])
    x_obs = np.asarray(dataset_split["x_obs"])
    mask = np.asarray(dataset_split["mask"])
    theta = np.asarray(dataset_split["theta"]) if "theta" in dataset_split else None
    indices = np.asarray(indices, dtype=int)

    if x_full.ndim != 2 or x_obs.shape != x_full.shape or mask.shape != x_full.shape:
        raise ValueError("x_full, x_obs, and mask must have shape (N, T).")
    if np.any(indices < 0) or np.any(indices >= x_full.shape[0]):
        raise IndexError("indices must be valid rows of dataset_split.")

    n_examples = int(indices.shape[0])
    if n_examples < 1:
        raise ValueError("indices must contain at least one example.")

    n_cols = min(3, n_examples)
    n_rows = math.ceil(n_examples / n_cols)
    if figsize is None:
        figsize = (5.0 * n_cols, 3.0 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    flat_axes = axes.ravel()

    for ax, index in zip(flat_axes, indices, strict=False):
        example_theta = None if theta is None else theta[index]
        plot_timeseries_example(
            ax,
            x_full[index],
            x_obs[index],
            mask[index],
            theta=example_theta,
            log_y=log_y,
        )
        ax.set_title(f"{split_name}[{index}]" + (f"\n{ax.get_title()}" if ax.get_title() else ""))

    for ax in flat_axes[n_examples:]:
        ax.set_visible(False)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_vector_example(
    ax: Any,
    x_full: np.ndarray,
    x_obs: np.ndarray,
    mask: np.ndarray,
    theta: np.ndarray | None = None,
) -> None:
    """Plot one complete, observed, and missing vector example."""
    x_full = np.asarray(x_full)
    x_obs = np.asarray(x_obs)
    mask = np.asarray(mask).astype(bool)

    if x_full.ndim != 1 or x_obs.shape != x_full.shape or mask.shape != x_full.shape:
        raise ValueError("x_full, x_obs, and mask must be 1D arrays with matching shapes.")

    feature = np.arange(x_full.shape[0])
    ax.plot(feature, x_full, color="0.75", linewidth=1.5, label="x_full", zorder=1)
    ax.scatter(feature[mask], x_obs[mask], color="C1", s=28, label="observed", zorder=3)

    missing = ~mask
    if np.any(missing):
        ax.scatter(
            feature[missing],
            x_full[missing],
            color="red",
            marker="x",
            s=36,
            label="missing",
            zorder=4,
        )

    theta_title = format_theta(theta)
    if theta_title:
        ax.set_title(theta_title)

    ax.set_xlabel("feature")
    ax.set_ylabel("x")
    ax.legend(fontsize="small")


def plot_vector_dataset_examples(
    dataset_split: dict[str, np.ndarray],
    indices: np.ndarray,
    output_path: str | os.PathLike[str],
    split_name: str = "train",
    figsize: tuple[float, float] | None = None,
) -> None:
    """Plot selected static vector examples from one dataset split and save them."""
    x_full = np.asarray(dataset_split["x_full"])
    x_obs = np.asarray(dataset_split["x_obs"])
    mask = np.asarray(dataset_split["mask"])
    theta = np.asarray(dataset_split["theta"]) if "theta" in dataset_split else None
    indices = np.asarray(indices, dtype=int)

    if x_full.ndim != 2 or x_obs.shape != x_full.shape or mask.shape != x_full.shape:
        raise ValueError("x_full, x_obs, and mask must have shape (N, D).")
    if np.any(indices < 0) or np.any(indices >= x_full.shape[0]):
        raise IndexError("indices must be valid rows of dataset_split.")

    n_examples = int(indices.shape[0])
    if n_examples < 1:
        raise ValueError("indices must contain at least one example.")

    n_cols = min(3, n_examples)
    n_rows = math.ceil(n_examples / n_cols)
    if figsize is None:
        figsize = (5.0 * n_cols, 3.0 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    flat_axes = axes.ravel()

    for ax, index in zip(flat_axes, indices, strict=False):
        example_theta = None if theta is None else theta[index]
        plot_vector_example(
            ax,
            x_full[index],
            x_obs[index],
            mask[index],
            theta=example_theta,
        )
        ax.set_title(f"{split_name}[{index}]" + (f"\n{ax.get_title()}" if ax.get_title() else ""))

    for ax in flat_axes[n_examples:]:
        ax.set_visible(False)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
