from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter


DEFAULT_INPUT = Path("outputs/analysis/full_experiment_results_with_diagnostics.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/analysis/figures/posterior_examples")

BUDGET_CLI = {
    "low": "low_sim_budget",
    "mid": "mid_sim_budget",
    "high": "high_sim_budget",
}
PROBLEM_ALIASES = {
    "oup": "oup",
    "glm": "glm",
    "glu": "glu",
    "lv": "lotka_volterra",
    "lotka_volterra": "lotka_volterra",
}
PROBLEM_STEMS = {
    "oup": "oup",
    "glm": "glm",
    "glu": "glu",
    "lotka_volterra": "lv",
}
PROBLEM_LABELS = {
    "oup": "OUP",
    "glm": "GLM",
    "glu": "GLU",
    "lotka_volterra": "Lotka-Volterra",
}
DEFAULT_PARAM_INDICES = {
    "oup": (0, 1),
    "glm": (2, 3),
    "glu": (8, 9),
    "lotka_volterra": (1, 3),
}
DEFAULT_REFERENCE_INDICES = {
    "oup": 0,
    "glm": 0,
    "glu": 0,
    "lotka_volterra": 4,
}
PARAMETER_MEANINGS = {
    "lotka_volterra": {
        0: "prey growth rate alpha",
        1: "predation rate beta",
        2: "predator death rate gamma",
        3: "predator reproduction rate delta",
    }
}
MISSINGNESS_ORDER = ["mcar", "mar", "mnar"]
MISSINGNESS_LABELS = {"mcar": "MCAR", "mar": "MAR", "mnar": "MNAR"}
METHOD_ORDER = [
    "npe_zero_imputation",
    "npe_mask_augmentation",
    "npe_learned_constant_imputation",
    "npe_probabilistic_learned_imputation",
    "npe_masked_transformer_embedding",
]
METHOD_LABELS = {
    "npe_zero_imputation": "Zero",
    "npe_mask_augmentation": "Zero + Mask",
    "npe_learned_constant_imputation": "Learned Constant",
    "npe_probabilistic_learned_imputation": "Probabilistic",
    "npe_masked_transformer_embedding": "Masked Transformer",
}
FRACTION_ORDER = [0.10, 0.25, 0.50]
FRACTION_LABELS = {0.10: "10%", 0.25: "25%", 0.50: "50%"}
COLORS = {
    "full": "#262626",
    0.10: "#30AFFF",
    0.25: "#FFAF00",
    0.50: "#FF0087",
}
@dataclass(frozen=True)
class PosteriorSet:
    label: str
    color: str
    samples: np.ndarray
    num_seeds: int


@dataclass(frozen=True)
class LoadReport:
    problem: str
    budget: str
    reference_idx: int
    param_indices: tuple[int, int]
    theta_true: np.ndarray
    missing_configs: list[str]
    loaded_configs: int
    samples_per_seed: int
    theta_dim: int
    x_dim: int | None
    axis_limits: list[tuple[float, float]]
    out_of_range_samples: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot qualitative posterior example overlays.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--problem", choices=sorted(PROBLEM_ALIASES), default="oup")
    parser.add_argument("--budget", choices=sorted(BUDGET_CLI), default="high")
    parser.add_argument("--reference-idx", type=int, default=None)
    parser.add_argument(
        "--samples-per-seed",
        type=int,
        default=1000,
        help="Maximum posterior samples to draw from each seed for each overlay.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on total samples per overlay after seed concatenation.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--param-indices",
        type=int,
        nargs=2,
        metavar=("I", "J"),
        help="One-based theta indices to plot, e.g. --param-indices 9 10.",
    )
    return parser.parse_args()


def normalize_epsilon(value: Any) -> float:
    if pd.isna(value):
        return math.nan
    epsilon = float(value)
    return epsilon / 100.0 if epsilon > 1.0 else epsilon


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = df.copy()
    out["epsilon"] = out["epsilon"].map(normalize_epsilon).round(2)
    for col in ["budget", "method", "problem", "missingness", "reference_posterior_samples_path"]:
        out[col] = out[col].astype(str)
    out["seed"] = out["seed"].astype(str)
    if "compile_status" in out:
        out = out[out["compile_status"].astype(str).str.lower() == "ok"]
    if "diagnostics_status" in out:
        out = out[out["diagnostics_status"].astype(str).str.lower() == "ok"]
    return out


def load_reference_samples(
    rows: pd.DataFrame,
    *,
    reference_idx: int,
    samples_per_seed: int,
    max_samples: int | None,
    param_indices: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[str], int, int | None]:
    samples_by_seed = []
    theta_true_by_seed = []
    missing_paths = []
    theta_dim = 0
    x_dim: int | None = None
    for _, row in rows.sort_values("seed").iterrows():
        path = Path(row["reference_posterior_samples_path"])
        if not path.exists():
            missing_paths.append(str(path))
            continue
        with h5py.File(path, "r") as h5:
            posterior = np.asarray(h5["theta_posterior"], dtype=float)
            theta_true = np.asarray(h5["theta_true"], dtype=float)
            theta_dim = int(posterior.shape[2])
            if "x_full" in h5:
                x_dim = int(np.prod(h5["x_full"].shape[1:]))
            if min(param_indices) < 0 or max(param_indices) >= theta_dim:
                one_based = tuple(idx + 1 for idx in param_indices)
                raise ValueError(
                    f"Selected theta indices {one_based} are invalid for {path}; "
                    f"theta_dim={theta_dim}."
                )
            if reference_idx < 0 or reference_idx >= posterior.shape[0]:
                raise ValueError(
                    f"reference_idx={reference_idx} out of range for {path}; "
                    f"available 0..{posterior.shape[0] - 1}."
                )
            seed_samples = posterior[reference_idx]
            if seed_samples.shape[0] > samples_per_seed:
                idx = rng.choice(seed_samples.shape[0], size=samples_per_seed, replace=False)
                seed_samples = seed_samples[idx]
            samples_by_seed.append(seed_samples[:, list(param_indices)])
            theta_true_by_seed.append(theta_true[reference_idx, list(param_indices)])

    if not samples_by_seed:
        return np.empty((0, 2)), np.full(2, np.nan), missing_paths, theta_dim, x_dim

    theta_true_stack = np.vstack(theta_true_by_seed)
    if not np.allclose(theta_true_stack, theta_true_stack[0], rtol=0, atol=1e-10):
        raise ValueError("theta_true differs across seed files for the selected reference index.")

    samples = np.vstack(samples_by_seed)
    if max_samples is not None and samples.shape[0] > max_samples:
        idx = rng.choice(samples.shape[0], size=max_samples, replace=False)
        samples = samples[idx]
    return samples, theta_true_stack[0], missing_paths, theta_dim, x_dim


def collect_posteriors(
    df: pd.DataFrame,
    *,
    problem: str,
    budget: str,
    reference_idx: int,
    param_indices: tuple[int, int],
    samples_per_seed: int,
    max_samples: int | None,
    seed: int,
) -> tuple[dict[tuple[str, str], list[PosteriorSet]], LoadReport]:
    rng = np.random.default_rng(seed)
    missing_configs: list[str] = []
    theta_true_values = []
    panels: dict[tuple[str, str], list[PosteriorSet]] = {}

    full_rows = df[
        (df["problem"] == problem)
        & (df["budget"] == budget)
        & (df["method"] == "npe_full_data")
        & (df["missingness"] == "none")
    ]
    full_samples, theta_true, missing_paths, theta_dim, x_dim = load_reference_samples(
        full_rows,
        reference_idx=reference_idx,
        samples_per_seed=samples_per_seed,
        max_samples=max_samples,
        param_indices=param_indices,
        rng=rng,
    )
    if len(full_rows) != 5 or full_samples.size == 0:
        missing_configs.append(f"full-data baseline: rows={len(full_rows)}, samples={full_samples.shape[0]}")
    missing_configs.extend(missing_paths)
    theta_true_values.append(theta_true)

    full = PosteriorSet(
        label="Full-data NPE",
        color=COLORS["full"],
        samples=full_samples,
        num_seeds=len(full_rows),
    )

    loaded_configs = 1 if full_samples.size else 0
    for missingness in MISSINGNESS_ORDER:
        for method in METHOD_ORDER:
            overlays = [full]
            for epsilon in FRACTION_ORDER:
                rows = df[
                    (df["problem"] == problem)
                    & (df["budget"] == budget)
                    & (df["method"] == method)
                    & (df["missingness"] == missingness)
                    & np.isclose(df["epsilon"].astype(float), epsilon)
                ]
                samples, theta, missing_paths, local_theta_dim, local_x_dim = load_reference_samples(
                    rows,
                    reference_idx=reference_idx,
                    samples_per_seed=samples_per_seed,
                    max_samples=max_samples,
                    param_indices=param_indices,
                    rng=rng,
                )
                theta_dim = max(theta_dim, local_theta_dim)
                if x_dim is None:
                    x_dim = local_x_dim
                if len(rows) != 5 or samples.size == 0:
                    missing_configs.append(
                        f"{method}/{missingness}/eps={epsilon:.2f}: rows={len(rows)}, samples={samples.shape[0]}"
                    )
                missing_configs.extend(missing_paths)
                if samples.size:
                    loaded_configs += 1
                    theta_true_values.append(theta)
                overlays.append(
                    PosteriorSet(
                        label=FRACTION_LABELS[epsilon],
                        color=COLORS[epsilon],
                        samples=samples,
                        num_seeds=len(rows),
                    )
                )
            panels[(missingness, method)] = overlays

    theta_stack = np.vstack([theta for theta in theta_true_values if np.all(np.isfinite(theta))])
    if theta_stack.size and not np.allclose(theta_stack, theta_stack[0], rtol=0, atol=1e-10):
        raise ValueError("theta_true differs across configurations for the selected reference index.")
    limits = global_limits(
        panels,
        theta_stack[0],
        problem=problem,
        param_indices=param_indices,
    )
    report = LoadReport(
        problem=problem,
        budget=budget,
        reference_idx=reference_idx,
        param_indices=param_indices,
        theta_true=theta_stack[0],
        missing_configs=missing_configs,
        loaded_configs=loaded_configs,
        samples_per_seed=samples_per_seed,
        theta_dim=theta_dim,
        x_dim=x_dim,
        axis_limits=limits,
        out_of_range_samples=count_out_of_range_samples(panels, limits),
    )
    return panels, report


def global_limits(
    panels: dict[tuple[str, str], list[PosteriorSet]],
    theta_true: np.ndarray,
    *,
    problem: str,
    param_indices: tuple[int, int],
) -> list[tuple[float, float]]:
    if problem == "glu":
        return [(-1.0, 1.0), (-1.0, 1.0)]
    if problem == "lotka_volterra":
        prior_log_mean = np.array([-0.125, -3.0, -0.125, -3.0], dtype=float)
        prior_log_std = np.full(4, 0.5, dtype=float)
        limits = []
        for idx in param_indices:
            lo, hi = np.exp(
                prior_log_mean[idx]
                + prior_log_std[idx] * np.array([-1.96, 1.96], dtype=float)
            )
            lo = min(float(lo), float(theta_true[len(limits)]))
            hi = max(float(hi), float(theta_true[len(limits)]))
            pad = 0.06 * (hi - lo)
            limits.append((float(max(0.0, lo - pad)), float(hi + pad)))
        return limits

    all_samples = []
    for overlays in panels.values():
        for posterior in overlays:
            if posterior.samples.size:
                all_samples.append(posterior.samples)
    samples = np.vstack(all_samples)
    limits = []
    quantiles = (0.02, 0.98) if problem == "glm" else (0.002, 0.998)
    for dim in range(2):
        lo, hi = np.quantile(samples[:, dim], quantiles)
        lo = min(lo, theta_true[dim])
        hi = max(hi, theta_true[dim])
        pad = 0.08 * (hi - lo) if hi > lo else 0.1
        limits.append((float(lo - pad), float(hi + pad)))
    return limits


def count_out_of_range_samples(
    panels: dict[tuple[str, str], list[PosteriorSet]],
    limits: list[tuple[float, float]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    xlim, ylim = limits
    for (missingness, method), overlays in panels.items():
        for posterior in overlays:
            if posterior.samples.size == 0:
                continue
            outside = (
                (posterior.samples[:, 0] < xlim[0])
                | (posterior.samples[:, 0] > xlim[1])
                | (posterior.samples[:, 1] < ylim[0])
                | (posterior.samples[:, 1] > ylim[1])
            )
            count = int(np.sum(outside))
            if count:
                key = f"{missingness}/{method}/{posterior.label}"
                counts[key] = count
    return counts


def plot_figure(
    panels: dict[tuple[str, str], list[PosteriorSet]],
    report: LoadReport,
    *,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    limits = report.axis_limits
    theta_labels = [theta_label(idx) for idx in report.param_indices]
    fig = plt.figure(figsize=(15.5, 9.2), constrained_layout=False)
    outer = fig.add_gridspec(
        len(MISSINGNESS_ORDER),
        len(METHOD_ORDER),
        left=0.055,
        right=0.995,
        bottom=0.075,
        top=0.835,
        wspace=0.18,
        hspace=0.28,
    )

    for row_idx, missingness in enumerate(MISSINGNESS_ORDER):
        for col_idx, method in enumerate(METHOD_ORDER):
            sub = outer[row_idx, col_idx].subgridspec(
                2,
                2,
                width_ratios=(2.0, 1.0),
                height_ratios=(1.0, 2.0),
                wspace=0.03,
                hspace=0.03,
            )
            ax_top = fig.add_subplot(sub[0, 0])
            ax_joint = fig.add_subplot(sub[1, 0], sharex=ax_top)
            ax_right = fig.add_subplot(sub[1, 1], sharey=ax_joint)
            ax_empty = fig.add_subplot(sub[0, 1])
            ax_empty.axis("off")

            if row_idx == 0:
                ax_top.set_title(METHOD_LABELS[method], fontsize=10, fontweight="bold", pad=10)
            if col_idx == 0:
                ax_joint.set_ylabel(MISSINGNESS_LABELS[missingness], fontsize=11, fontweight="bold")

            draw_corner_panel(
                ax_joint,
                ax_top,
                ax_right,
                panels[(missingness, method)],
                theta_true=report.theta_true,
                limits=limits,
            )
            format_panel_axes(ax_joint, ax_top, ax_right, row_idx=row_idx, col_idx=col_idx)

    fig.suptitle(
        (
            f"{PROBLEM_LABELS[report.problem]} posterior example, reference observation "
            f"{report.reference_idx} - marginal over ({theta_labels[0]}, {theta_labels[1]})"
        ),
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    handles = [
        Line2D([0], [0], color=COLORS["full"], linewidth=1.3, label="Full-data NPE"),
        *[
            Line2D([0], [0], color=COLORS[epsilon], linewidth=1.3, label=FRACTION_LABELS[epsilon])
            for epsilon in FRACTION_ORDER
        ],
        Line2D([0], [0], color="#111111", marker="s", linewidth=0, markersize=5, label="true theta"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=5,
        frameon=False,
        fontsize=9,
        handlelength=2.0,
        columnspacing=1.3,
    )
    fig.supxlabel(theta_labels[0], fontsize=11, y=0.025)
    fig.supylabel(theta_labels[1], fontsize=11, x=0.012)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{PROBLEM_STEMS[report.problem]}_posterior_example_ref{report.reference_idx}"
    paths = [output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return paths


def draw_corner_panel(
    ax_joint: plt.Axes,
    ax_top: plt.Axes,
    ax_right: plt.Axes,
    overlays: list[PosteriorSet],
    *,
    theta_true: np.ndarray,
    limits: list[tuple[float, float]],
) -> None:
    for posterior in overlays:
        if posterior.samples.size == 0:
            continue
        alpha = 0.95 if posterior.label == "Full-data NPE" else 0.82
        linewidth = 1.15 if posterior.label == "Full-data NPE" else 1.0
        draw_1d_density(ax_top, posterior.samples[:, 0], color=posterior.color, limits=limits[0], alpha=alpha, linewidth=linewidth)
        draw_1d_density(
            ax_right,
            posterior.samples[:, 1],
            color=posterior.color,
            limits=limits[1],
            alpha=alpha,
            linewidth=linewidth,
            orientation="horizontal",
        )
        draw_2d_contours(
            ax_joint,
            posterior.samples[:, 0],
            posterior.samples[:, 1],
            color=posterior.color,
            xlim=limits[0],
            ylim=limits[1],
            alpha=alpha,
            linewidth=linewidth,
        )

    ax_joint.axvline(theta_true[0], color="#111111", linestyle=(0, (3, 3)), linewidth=0.9, zorder=1)
    ax_joint.axhline(theta_true[1], color="#111111", linestyle=(0, (3, 3)), linewidth=0.9, zorder=1)
    ax_top.axvline(theta_true[0], color="#111111", linestyle=(0, (3, 3)), linewidth=0.9)
    ax_right.axhline(theta_true[1], color="#111111", linestyle=(0, (3, 3)), linewidth=0.9)
    ax_joint.scatter(theta_true[0], theta_true[1], color="#111111", marker="s", s=22, zorder=6)


def draw_1d_density(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    color: str,
    limits: tuple[float, float],
    alpha: float,
    linewidth: float,
    orientation: str = "vertical",
) -> None:
    hist, edges = np.histogram(values, bins=45, range=limits, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if orientation == "vertical":
        ax.plot(centers, hist, color=color, alpha=alpha, linewidth=linewidth)
        ax.fill_between(centers, 0, hist, color=color, alpha=0.035)
    else:
        ax.plot(hist, centers, color=color, alpha=alpha, linewidth=linewidth)
        ax.fill_betweenx(centers, 0, hist, color=color, alpha=0.035)


def draw_2d_contours(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    alpha: float,
    linewidth: float,
) -> None:
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=70, range=[xlim, ylim])
    hist = gaussian_filter(hist, sigma=1.15)
    levels = credible_levels(hist, probs=(0.95, 0.68))
    if levels.size == 0:
        return
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    ax.contour(
        x_centers,
        y_centers,
        hist.T,
        levels=levels,
        colors=[color] * len(levels),
        linewidths=[linewidth * 0.75, linewidth],
        alpha=alpha,
        zorder=3,
    )


def credible_levels(hist: np.ndarray, probs: tuple[float, ...]) -> np.ndarray:
    flat = np.sort(hist.ravel())[::-1]
    total = flat.sum()
    if total <= 0:
        return np.array([], dtype=float)
    cdf = np.cumsum(flat) / total
    levels = []
    for prob in probs:
        idx = min(np.searchsorted(cdf, prob), flat.size - 1)
        level = flat[idx]
        if level > 0:
            levels.append(level)
    return np.unique(np.sort(levels))


def format_panel_axes(
    ax_joint: plt.Axes,
    ax_top: plt.Axes,
    ax_right: plt.Axes,
    *,
    row_idx: int,
    col_idx: int,
) -> None:
    for ax in [ax_joint, ax_top, ax_right]:
        ax.set_facecolor("white")
        ax.grid(axis="both", color="#ececec", linewidth=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#b8b8b8")
        ax.spines["bottom"].set_color("#b8b8b8")
        ax.tick_params(axis="both", labelsize=6.5, length=2.0, color="#777777")

    ax_top.tick_params(labelbottom=False, labelleft=False)
    ax_right.tick_params(labelbottom=False, labelleft=False)
    ax_top.set_yticks([])
    ax_right.set_xticks([])

    if row_idx < len(MISSINGNESS_ORDER) - 1:
        ax_joint.tick_params(labelbottom=False)
    if col_idx > 0:
        ax_joint.tick_params(labelleft=False)
    if row_idx == len(MISSINGNESS_ORDER) - 1:
        ax_joint.set_xlabel("")
    if col_idx == 0:
        ax_joint.set_ylabel(ax_joint.get_ylabel())


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.7,
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def theta_label(index: int) -> str:
    return rf"$\theta_{{{index + 1}}}$"


def parameter_meaning(problem: str, index: int) -> str:
    return PARAMETER_MEANINGS.get(problem, {}).get(index, f"theta_{index + 1}")


def write_pair_selection_report(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "parameter_pair_selection.md"
    text = """# Posterior Example Parameter Pair Selection

## GLM

- Simulator structure: Bernoulli GLM with `theta_dim=10`. Reference observations are raw Bernoulli time series with `x_dim=100`; benchmark posterior files store 10-dimensional theta samples.
- Parameter/observation mapping: theta enters through a fixed design matrix. `theta_1` is an intercept and `theta_2`-`theta_10` are stimulus-lag weights. Individual theta dimensions do not correspond directly to individual observation coordinates.
- MAR behavior: `coordinate_mar`, `mode="increasing"`; missing probability increases monotonically with raw observation coordinate/time. This preferentially removes late time points, but does not imply that the last theta dimensions are the most affected.
- MNAR behavior: mean-normalized self-censoring; missing probability depends on per-observation min/max-normalized values. For binary GLM observations this preferentially censors spikes (`x=1`) relative to non-spikes (`x=0`) and has no inherent first-vs-last coordinate asymmetry.
- Lightweight posterior diagnostic: stored high-budget posterior samples for reference observation 0 showed strong MAR/MNAR degradation in middle lag dimensions, especially `theta_3` and `theta_4`.
- Recommended pair: `(theta_3, theta_4)`.

## GLU

- Simulator structure: Gaussian Linear Uniform with `theta_dim=10` and `x_dim=10`.
- Parameter/observation mapping: coordinate-wise, `x_i = theta_i + Normal(0, 0.1)`.
- MAR behavior: `coordinate_mar`, `mode="increasing"`; missing probability increases by coordinate, so later observation coordinates are preferentially masked. Because GLU is coordinate-wise, this directly removes information about later theta dimensions.
- MNAR behavior: self-censoring based on per-observation min/max-normalized realized `x_i` values. It has no fixed first-vs-last coordinate asymmetry, but high-valued coordinates are preferentially censored within each observation.
- Lightweight posterior diagnostic: stored high-budget posterior samples showed the largest MAR degradation for `theta_9` and `theta_10`; these dimensions also remain informative under MNAR.
- Recommended pair: `(theta_9, theta_10)`.

## Lotka-Volterra

- Simulator structure: SBIBM-inspired Lotka-Volterra ODE with `theta_dim=4` and `x_dim=100`, corresponding to 50 time points with interleaved prey and predator observations.
- Parameter meanings: `theta_1 = alpha` prey growth, `theta_2 = beta` predation, `theta_3 = gamma` predator death, and `theta_4 = delta` predator reproduction.
- Observation structure: the simulator solves prey/predator dynamics from initial state `(30, 1)` over 20 days and observes both populations with lognormal noise.
- MAR behavior: `lv_time_mar`, `mode="increasing"`; missing probability increases by time index and hides both populations at the selected time slice. This preferentially removes late-time dynamical information rather than one flattened population coordinate.
- MNAR behavior: `lv_log_total_mnar`; missing probability depends on mean-normalized per-sample min/max scores of log total population. It hides both populations at high-total-population time slices and has no direct one-parameter asymmetry.
- Lightweight posterior diagnostic: full-data high-budget posteriors are already imperfect on LV. For reference observation 0, `(theta_2, theta_4)` is an interpretable interaction-rate pair; both dimensions retain full-data constraining power and show meaningful variance/error changes under missingness. Across full-data diagnostics, reference 0 is relatively easy by C2ST, so the automatic LV example uses reference observation 4, the closest median-difficulty reference under a fixed criterion combining full-data C2ST, posterior mean shift, and covariance trace ratio.
- Recommended pair: `(theta_2, theta_4)`.
"""
    path.write_text(text, encoding="utf-8")
    return path


def write_captions(output_dir: Path, report: LoadReport, paths: list[Path]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "captions.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Posterior Example Captions\n"
    stem = paths[0].stem
    theta_labels = [theta_label(idx).replace("$", "") for idx in report.param_indices]
    caption = (
        f"{PROBLEM_LABELS[report.problem]} posterior example for reference observation "
        f"{report.reference_idx}, showing only the selected 2D marginal over "
        f"({theta_labels[0]}, {theta_labels[1]}) of the full {report.theta_dim}-dimensional "
        f"posterior. The selected true parameter values are "
        f"({report.theta_true[0]:.6g}, {report.theta_true[1]:.6g}). The simulation budget is "
        f"{report.budget}. Rows are missingness mechanisms and columns are inference strategies. "
        "Each panel overlays Full-Data NPE in black with 10%, 25%, and 50% missing fractions in "
        "blue, orange, and magenta. Posterior samples are concatenated across five independently "
        f"trained seeds, using up to {report.samples_per_seed} samples per seed. Contours show "
        "68% and 95% posterior mass; dashed lines and the black square mark the true selected "
        "parameter values. Axis limits are fixed across all panels for the problem."
    )
    if report.problem == "lotka_volterra":
        caption += (
            " Lotka-Volterra is a difficult stress-test problem in this benchmark; the Full-Data "
            "NPE overlay is a reference estimator trained on complete observations, not ground truth, "
            "and is itself not necessarily an accurate posterior approximation."
        )
    if report.out_of_range_samples:
        caption += (
            " Some posterior samples lie outside the displayed fixed parameter range; axis limits were "
            "kept fixed for readability and comparability, and this should be interpreted as broad or "
            "pathological posterior mass rather than discarded scientific evidence."
        )
    block = f"\n## `{stem}`\n\n{caption}\n"
    marker = f"## `{stem}`"
    if marker in existing:
        prefix = existing.split(marker)[0].rstrip()
        suffix = existing.split(marker, 1)[1]
        rest = suffix.split("\n## `", 1)
        tail = ("\n## `" + rest[1]) if len(rest) == 2 else ""
        text = prefix + block + tail
    else:
        text = existing.rstrip() + "\n" + block
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    budget = BUDGET_CLI[args.budget]
    problem = PROBLEM_ALIASES[args.problem]
    reference_idx = (
        DEFAULT_REFERENCE_INDICES[problem]
        if args.reference_idx is None
        else args.reference_idx
    )
    param_indices = (
        tuple(idx - 1 for idx in args.param_indices)
        if args.param_indices is not None
        else DEFAULT_PARAM_INDICES[problem]
    )
    if param_indices[0] == param_indices[1]:
        raise SystemExit("--param-indices must select two distinct theta dimensions.")
    df = load_results(args.input)
    panels, report = collect_posteriors(
        df,
        problem=problem,
        budget=budget,
        reference_idx=reference_idx,
        param_indices=param_indices,
        samples_per_seed=args.samples_per_seed,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    paths = plot_figure(panels, report, output_dir=args.output_dir, dpi=args.dpi)
    pair_report_path = write_pair_selection_report(args.output_dir)
    captions_path = write_captions(args.output_dir, report, paths)

    print("Posterior example validation")
    print(f"Problem: {report.problem}")
    print(f"Budget: {report.budget}")
    print(f"Theta dimension: {report.theta_dim}")
    print(f"x dimension: {report.x_dim}")
    print(f"Selected parameter pair: ({report.param_indices[0] + 1}, {report.param_indices[1] + 1})")
    print(
        "Selected parameter meanings: "
        f"{parameter_meaning(report.problem, report.param_indices[0])}; "
        f"{parameter_meaning(report.problem, report.param_indices[1])}"
    )
    print(f"Reference observation index: {report.reference_idx}")
    print(f"True parameter value: [{report.theta_true[0]:.6g}, {report.theta_true[1]:.6g}]")
    print(
        "Aggregation: concatenated posterior samples across the five seeds; "
        f"up to {report.samples_per_seed} samples per seed per overlay."
    )
    print(f"Loaded overlay configurations, including full-data baseline: {report.loaded_configs}")
    print(f"Missing configurations/files: {len(report.missing_configs)}")
    for item in report.missing_configs:
        print(f"  {item}")
    print(f"Samples outside fixed plotting limits: {sum(report.out_of_range_samples.values())}")
    for key, value in sorted(report.out_of_range_samples.items()):
        print(f"  {key}: {value}")
    print("Wrote posterior example figures:")
    for path in paths:
        print(f"  {path}")
    print(f"Wrote parameter-pair report: {pair_report_path}")
    print(f"Wrote captions: {captions_path}")


if __name__ == "__main__":
    main()
