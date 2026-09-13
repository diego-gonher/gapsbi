from __future__ import annotations

import argparse
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


DEFAULT_INPUT = Path("outputs/analysis/full_experiment_results_with_diagnostics.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/analysis/figures/benchmark")

BUDGET_ORDER = ["low_sim_budget", "mid_sim_budget", "high_sim_budget"]
BUDGET_LABELS = {
    "low_sim_budget": r"$10^3$",
    "mid_sim_budget": r"$10^4$",
    "high_sim_budget": r"$10^5$",
}
PROBLEM_ORDER = ["oup", "glm", "glu", "lotka_volterra"]
PROBLEM_LABELS = {
    "oup": "OUP",
    "glm": "GLM",
    "glu": "GLU",
    "lotka_volterra": "Lotka-Volterra",
}
METHOD_ORDER = [
    "npe_full_data",
    "npe_zero_imputation",
    "npe_mask_augmentation",
    "npe_learned_constant_imputation",
    "npe_probabilistic_learned_imputation",
    "npe_masked_transformer_embedding",
]
METHOD_LABELS = {
    "npe_full_data": "Full-Data NPE",
    "npe_zero_imputation": "Zero",
    "npe_mask_augmentation": "Zero +\nMask",
    "npe_learned_constant_imputation": "Learned\nConstant",
    "npe_probabilistic_learned_imputation": "Probabilistic",
    "npe_masked_transformer_embedding": "Masked\nTransformer",
}
MISSINGNESS_ORDER = ["mcar", "mar", "mnar"]
MISSINGNESS_LABELS = {
    "mcar": "MCAR",
    "mar": "MAR",
    "mnar": "MNAR",
}
COMBINED_MISSINGNESS_ORDER = ["mcar", "mnar"]
COMBINED_MISSINGNESS_LINESTYLES = {
    "mcar": "-",
    "mnar": (0, (4, 2.5)),
}
COMBINED_MISSINGNESS_OFFSCALE_OPEN = {
    "mcar": False,
    "mnar": True,
}
FRACTION_ORDER = [0.10, 0.25, 0.50]
FRACTION_LABELS = {
    0.10: "10%",
    0.25: "25%",
    0.50: "50%",
}
FRACTION_COLORS = {
    0.10: "#30AFFF",
    0.25: "#FFAF00",
    0.50: "#FF0087",
}
FULL_DATA_COLOR = "#262626"
FULL_DATA_MARKER = "*"
OFFSCALE_X_DELTA = 0.055
OFFSCALE_X_OFFSETS = {
    0.10: -OFFSCALE_X_DELTA,
    0.25: 0.0,
    0.50: OFFSCALE_X_DELTA,
}
OFFSCALE_MARKER_SIZE = 24
OFFSCALE_MARKER_ALPHA = 0.55
BASE_MARKER_SIZE = 3.1
COMBINED_MARKER_SIZE = 2.8
FULL_DATA_BASE_MARKER_SIZE = 5.0
FULL_DATA_COMBINED_MARKER_SIZE = 4.8


@dataclass(frozen=True)
class MetricSpec:
    key: str
    column: str
    label: str
    filename_prefix: str
    ideal: float | None = None
    yscale: str = "linear"
    ylim: tuple[float, float] | None = None


METRICS = {
    "c2st": MetricSpec(
        key="c2st",
        column="reference_c2st_accuracy_mean",
        label="C2ST accuracy",
        filename_prefix="c2st",
        ideal=0.5,
        ylim=(0.5, 1.0),
    ),
    "tarp_mae": MetricSpec(
        key="tarp_mae",
        column="tarp_mae",
        label="TARP MAE",
        filename_prefix="tarp_mae",
        ideal=0.0,
        ylim=(0.0, 0.10),
    ),
    "mean_shift": MetricSpec(
        key="mean_shift",
        column="reference_posterior_mean_shift_median",
        label="Posterior mean shift",
        filename_prefix="mean_shift",
        ideal=0.0,
        ylim=(0.0, 1.5),
    ),
    "covariance_trace_ratio": MetricSpec(
        key="covariance_trace_ratio",
        column="reference_covariance_trace_ratio_median",
        label="Covariance trace ratio",
        filename_prefix="covariance_trace_ratio",
        ideal=1.0,
        yscale="log",
        ylim=(0.8, 1000.0),
    ),
}


@dataclass(frozen=True)
class FigureReport:
    stem: str
    metric_key: str
    missingness: str
    display_range: tuple[float, float]
    offscale_points: int
    max_actual: float
    yscale: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SBIBM-style benchmark summary figures.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metric", choices=sorted(METRICS), help="Metric to plot.")
    parser.add_argument(
        "--missingness",
        choices=MISSINGNESS_ORDER,
        help="Missingness mechanism to plot.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all metric/missingness combinations.",
    )
    parser.add_argument(
        "--combined-mcar-mnar",
        action="store_true",
        help=(
            "Generate combined MCAR/MNAR figures. MCAR uses solid lines; "
            "MNAR uses dashed lines and open off-scale triangles."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("pdf", "png"),
        action="append",
        dest="formats",
        help="Output format. Repeat to save multiple. Default: pdf and png.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_metadata(df)
    return df


def normalize_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["epsilon"] = out["epsilon"].map(normalize_epsilon)
    out["budget"] = out["budget"].astype(str)
    out["method"] = out["method"].astype(str)
    out["problem"] = out["problem"].astype(str)
    out["missingness"] = out["missingness"].astype(str)
    out["seed"] = out["seed"].astype(str)
    return out


def normalize_epsilon(value: Any) -> float:
    if pd.isna(value):
        return math.nan
    epsilon = float(value)
    return epsilon / 100.0 if epsilon > 1.0 else epsilon


def validate_results(df: pd.DataFrame) -> None:
    expected_methods = set(METHOD_ORDER)
    expected_problems = set(PROBLEM_ORDER)
    expected_budgets = set(BUDGET_ORDER)
    expected_missingness = set(MISSINGNESS_ORDER) | {"none"}

    methods = set(df["method"].unique())
    problems = set(df["problem"].unique())
    budgets = set(df["budget"].unique())
    missingness = set(df["missingness"].unique())

    missing_methods = expected_methods - methods
    missing_problems = expected_problems - problems
    missing_budgets = expected_budgets - budgets
    missing_mechanisms = expected_missingness - missingness
    if missing_methods or missing_problems or missing_budgets or missing_mechanisms:
        raise ValueError(
            "Missing expected metadata: "
            f"methods={sorted(missing_methods)}, "
            f"problems={sorted(missing_problems)}, "
            f"budgets={sorted(missing_budgets)}, "
            f"missingness={sorted(missing_mechanisms)}"
        )

    missing_methods_df = df[df["method"] != "npe_full_data"]
    fractions = sorted(
        {
            round(float(value), 2)
            for value in missing_methods_df["epsilon"].dropna().unique()
        }
    )
    expected_fractions = sorted(FRACTION_ORDER)
    if fractions != expected_fractions:
        raise ValueError(
            f"Missing-data methods contain unexpected fractions: {fractions}; "
            f"expected {expected_fractions}."
        )

    config_counts = (
        df.groupby(["budget", "method", "problem", "missingness", "epsilon"], dropna=False)
        .agg(num_rows=("seed", "count"), num_seeds=("seed", "nunique"))
        .reset_index()
    )
    failed = df[
        (df.get("compile_status", "ok") != "ok")
        | (df.get("diagnostics_status", "ok") != "ok")
    ]
    incomplete = config_counts[config_counts["num_seeds"] < 5]

    print("Benchmark plotting validation")
    print(f"Input rows: {len(df)}")
    print(f"Methods found: {', '.join(sorted(methods))}")
    print(f"Problems found: {', '.join(PROBLEM_LABELS[p] for p in PROBLEM_ORDER)}")
    print(f"Budgets found: {', '.join(BUDGET_LABELS[b] for b in BUDGET_ORDER)}")
    print(f"Missingness found: {', '.join(sorted(missingness))}")
    print(f"Normalized missing fractions found: {', '.join(FRACTION_LABELS[f] for f in FRACTION_ORDER)}")
    print("Seed counts per configuration:")
    print(config_counts["num_seeds"].value_counts().sort_index().to_string())
    print(f"Configurations with failed rows: {failed.shape[0]}")
    print(f"Configurations with fewer than 5 seeds: {len(incomplete)}")
    if len(incomplete):
        print(incomplete.to_string(index=False))


def successful_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "compile_status" in out:
        out = out[out["compile_status"].astype(str).str.lower() == "ok"]
    if "diagnostics_status" in out:
        out = out[out["diagnostics_status"].astype(str).str.lower() == "ok"]
    return out


def prepare_panel_data(
    df: pd.DataFrame,
    *,
    metric: MetricSpec,
    method: str,
    problem: str,
    missingness: str,
    epsilon: float | None,
) -> pd.DataFrame:
    rows = df[(df["method"] == method) & (df["problem"] == problem)]
    if method == "npe_full_data":
        rows = rows[rows["missingness"] == "none"]
    else:
        rows = rows[
            (rows["missingness"] == missingness)
            & np.isclose(rows["epsilon"].astype(float), float(epsilon))
        ]
    if metric.column not in rows.columns:
        raise ValueError(f"Missing metric column: {metric.column}")

    rows = rows[rows[metric.column].notna()].copy()
    rows[metric.column] = pd.to_numeric(rows[metric.column], errors="coerce")
    rows = rows[np.isfinite(rows[metric.column])]

    grouped = (
        rows.groupby("budget")
        .agg(
            mean=(metric.column, "mean"),
            std=(metric.column, "std"),
            n=(metric.column, "count"),
        )
        .reindex(BUDGET_ORDER)
        .reset_index()
    )
    grouped["se"] = grouped["std"] / np.sqrt(grouped["n"])
    grouped["ci95"] = 1.96 * grouped["se"]
    grouped.loc[grouped["n"] <= 1, "ci95"] = 0.0
    return grouped


def metric_limits(metric: MetricSpec) -> tuple[float, float]:
    if metric.ylim is None:
        raise ValueError(f"Metric {metric.key} must define a common display range.")
    return metric.ylim


def plot_benchmark_grid(
    df: pd.DataFrame,
    *,
    metric: MetricSpec,
    missingness: str,
) -> tuple[plt.Figure, FigureReport]:
    fig, axes = plt.subplots(
        nrows=len(PROBLEM_ORDER),
        ncols=len(METHOD_ORDER),
        figsize=(13.2, 8.4),
        sharex=True,
        constrained_layout=False,
    )
    x = np.arange(len(BUDGET_ORDER))
    y_limits = metric_limits(metric)
    offscale_points = 0
    max_actual = -math.inf

    for row_idx, problem in enumerate(PROBLEM_ORDER):
        for col_idx, method in enumerate(METHOD_ORDER):
            ax = axes[row_idx, col_idx]
            style_axis(ax, metric, y_limits)
            if row_idx == 0:
                ax.set_title(METHOD_LABELS[method], fontsize=10.5, fontweight="bold", pad=8)
            if col_idx == 0:
                ax.set_ylabel(PROBLEM_LABELS[problem], fontsize=11.5, fontweight="bold")

            if method == "npe_full_data":
                panel = prepare_panel_data(
                    df,
                    metric=metric,
                    method=method,
                    problem=problem,
                    missingness=missingness,
                    epsilon=None,
                )
                curve_offscale, curve_max = plot_curve(
                    ax,
                    x,
                    panel,
                    color=FULL_DATA_COLOR,
                    label="Full-data",
                    y_limits=y_limits,
                    offscale_x_offset=0.0,
                    linestyle="-",
                    offscale_open=False,
                    marker_size=FULL_DATA_BASE_MARKER_SIZE,
                    marker=FULL_DATA_MARKER,
                    marker_open=False,
                )
                offscale_points += curve_offscale
                max_actual = max(max_actual, curve_max)
            else:
                for epsilon in FRACTION_ORDER:
                    panel = prepare_panel_data(
                        df,
                        metric=metric,
                        method=method,
                        problem=problem,
                        missingness=missingness,
                        epsilon=epsilon,
                    )
                    curve_offscale, curve_max = plot_curve(
                        ax,
                        x,
                        panel,
                        color=FRACTION_COLORS[epsilon],
                        label=FRACTION_LABELS[epsilon],
                        y_limits=y_limits,
                        offscale_x_offset=OFFSCALE_X_OFFSETS[epsilon],
                        linestyle="-",
                        offscale_open=False,
                        marker_size=BASE_MARKER_SIZE,
                        marker_open=False,
                    )
                    offscale_points += curve_offscale
                    max_actual = max(max_actual, curve_max)

            if row_idx < len(PROBLEM_ORDER) - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xticks(x, [BUDGET_LABELS[b] for b in BUDGET_ORDER], fontsize=9.8)
            if col_idx > 0:
                ax.tick_params(labelleft=False)

    fig.suptitle(
        f"{metric.label} under {MISSINGNESS_LABELS[missingness]} missingness",
        fontsize=14,
        fontweight="bold",
        y=0.988,
    )
    fig.supxlabel("Simulation budget", fontsize=12, y=0.040)
    fig.supylabel(metric.label, fontsize=12, x=0.012)

    handles = [
        Line2D([0], [0], color=FULL_DATA_COLOR, marker=FULL_DATA_MARKER, linewidth=1.0, markersize=5.0, label="Full-data NPE"),
        *[
            Line2D(
                [0],
                [0],
                color=FRACTION_COLORS[epsilon],
                marker="o",
                linewidth=1.0,
                markersize=3.5,
                label=FRACTION_LABELS[epsilon],
            )
            for epsilon in FRACTION_ORDER
        ],
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
        fontsize=10,
        handlelength=2.0,
        columnspacing=1.5,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.10, top=0.825, wspace=0.10, hspace=0.16)
    report = FigureReport(
        stem=f"{metric.filename_prefix}_{missingness}",
        metric_key=metric.key,
        missingness=missingness,
        display_range=y_limits,
        offscale_points=offscale_points,
        max_actual=max_actual if np.isfinite(max_actual) else math.nan,
        yscale=metric.yscale,
    )
    return fig, report


def plot_combined_missingness_grid(
    df: pd.DataFrame,
    *,
    metric: MetricSpec,
) -> tuple[plt.Figure, FigureReport]:
    fig, axes = plt.subplots(
        nrows=len(PROBLEM_ORDER),
        ncols=len(METHOD_ORDER),
        figsize=(13.2, 8.4),
        sharex=True,
        constrained_layout=False,
    )
    x = np.arange(len(BUDGET_ORDER))
    y_limits = metric_limits(metric)
    offscale_points = 0
    max_actual = -math.inf

    for row_idx, problem in enumerate(PROBLEM_ORDER):
        for col_idx, method in enumerate(METHOD_ORDER):
            ax = axes[row_idx, col_idx]
            style_axis(ax, metric, y_limits)
            if row_idx == 0:
                ax.set_title(METHOD_LABELS[method], fontsize=10.5, fontweight="bold", pad=8)
            if col_idx == 0:
                ax.set_ylabel(PROBLEM_LABELS[problem], fontsize=11.5, fontweight="bold")

            if method == "npe_full_data":
                panel = prepare_panel_data(
                    df,
                    metric=metric,
                    method=method,
                    problem=problem,
                    missingness="mcar",
                    epsilon=None,
                )
                curve_offscale, curve_max = plot_curve(
                    ax,
                    x,
                    panel,
                    color=FULL_DATA_COLOR,
                    label="Full-data",
                    y_limits=y_limits,
                    offscale_x_offset=0.0,
                    linestyle="-",
                    offscale_open=False,
                    marker_size=FULL_DATA_COMBINED_MARKER_SIZE,
                    marker=FULL_DATA_MARKER,
                    marker_open=False,
                )
                offscale_points += curve_offscale
                max_actual = max(max_actual, curve_max)
            else:
                for missingness in COMBINED_MISSINGNESS_ORDER:
                    for epsilon in FRACTION_ORDER:
                        panel = prepare_panel_data(
                            df,
                            metric=metric,
                            method=method,
                            problem=problem,
                            missingness=missingness,
                            epsilon=epsilon,
                        )
                        curve_offscale, curve_max = plot_curve(
                            ax,
                            x,
                            panel,
                            color=FRACTION_COLORS[epsilon],
                            label=f"{MISSINGNESS_LABELS[missingness]} {FRACTION_LABELS[epsilon]}",
                            y_limits=y_limits,
                            offscale_x_offset=OFFSCALE_X_OFFSETS[epsilon],
                            linestyle=COMBINED_MISSINGNESS_LINESTYLES[missingness],
                            offscale_open=COMBINED_MISSINGNESS_OFFSCALE_OPEN[missingness],
                            marker_size=COMBINED_MARKER_SIZE,
                            marker_open=missingness == "mnar",
                        )
                        offscale_points += curve_offscale
                        max_actual = max(max_actual, curve_max)

            if row_idx < len(PROBLEM_ORDER) - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xticks(x, [BUDGET_LABELS[b] for b in BUDGET_ORDER], fontsize=9.8)
            if col_idx > 0:
                ax.tick_params(labelleft=False)

    fig.suptitle(
        f"{metric.label} under MCAR and MNAR missingness",
        fontsize=14,
        fontweight="bold",
        y=0.988,
    )
    fig.supxlabel("Simulation budget", fontsize=12, y=0.040)
    fig.supylabel(metric.label, fontsize=12, x=0.012)
    fig.legend(
        handles=combined_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=6,
        frameon=False,
        fontsize=9.7,
        handlelength=2.0,
        columnspacing=1.15,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.10, top=0.825, wspace=0.10, hspace=0.16)
    report = FigureReport(
        stem=f"{metric.filename_prefix}_mcar_mnar",
        metric_key=metric.key,
        missingness="mcar_mnar",
        display_range=y_limits,
        offscale_points=offscale_points,
        max_actual=max_actual if np.isfinite(max_actual) else math.nan,
        yscale=metric.yscale,
    )
    return fig, report


def combined_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=FULL_DATA_COLOR, marker=FULL_DATA_MARKER, linewidth=1.0, markersize=4.8, label="Full-data NPE"),
        *[
            Line2D(
                [0],
                [0],
                color=FRACTION_COLORS[epsilon],
                marker="o",
                linewidth=1.0,
                markersize=3.2,
                label=FRACTION_LABELS[epsilon],
            )
            for epsilon in FRACTION_ORDER
        ],
        Line2D([0], [0], color="#4a4a4a", linewidth=1.05, linestyle="-", marker="o", markersize=3.2, label="MCAR"),
        Line2D(
            [0],
            [0],
            color="#4a4a4a",
            linewidth=1.05,
            linestyle=COMBINED_MISSINGNESS_LINESTYLES["mnar"],
            marker="o",
            markerfacecolor="none",
            markeredgecolor="#4a4a4a",
            markersize=3.2,
            label="MNAR",
        ),
    ]


def style_axis(
    ax: plt.Axes,
    metric: MetricSpec,
    y_limits: tuple[float, float] | None,
) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.55)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b8b8b8")
    ax.spines["bottom"].set_color("#b8b8b8")
    ax.tick_params(axis="both", labelsize=9, length=2.5, color="#777777")
    ax.set_yscale(metric.yscale)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    if metric.ideal is not None:
        ax.axhline(metric.ideal, color="#9a9a9a", linewidth=0.7, linestyle=(0, (3, 3)), zorder=0)


def plot_curve(
    ax: plt.Axes,
    x: np.ndarray,
    panel: pd.DataFrame,
    *,
    color: str,
    label: str,
    y_limits: tuple[float, float],
    offscale_x_offset: float,
    linestyle: str | tuple[int, tuple[float, ...]] = "-",
    offscale_open: bool = False,
    marker_size: float = BASE_MARKER_SIZE,
    marker: str = "o",
    marker_open: bool = False,
) -> tuple[int, float]:
    means = panel["mean"].to_numpy(dtype=float)
    ci95 = panel["ci95"].fillna(0.0).to_numpy(dtype=float)
    valid = np.isfinite(means)
    if not np.any(valid):
        return 0, math.nan

    lower, upper = y_limits
    in_range = valid & (means <= upper) & (means >= lower)
    offscale = valid & (means > upper)
    marker_face_color = "none" if marker_open else color

    for segment in contiguous_segments(np.where(in_range)[0]):
        ax.errorbar(
            x[segment],
            means[segment],
            yerr=ci95[segment],
            color=color,
            marker=marker,
            markersize=marker_size,
            markerfacecolor=marker_face_color,
            markeredgecolor=color,
            markeredgewidth=0.8 if marker_open else 0.0,
            linewidth=1.0,
            linestyle=linestyle,
            elinewidth=0.55,
            capsize=1.5,
            capthick=0.55,
            alpha=0.95,
            label=label,
        )

    if np.any(offscale):
        facecolors = "none" if offscale_open else color
        edgecolors = color if offscale_open else "white"
        linewidths = 0.9 if offscale_open else 0.45
        ax.scatter(
            x[offscale] + offscale_x_offset,
            np.full(np.count_nonzero(offscale), offscale_marker_y(ax, upper)),
            marker="^",
            s=OFFSCALE_MARKER_SIZE,
            facecolors=facecolors,
            edgecolors=edgecolors,
            linewidths=linewidths,
            alpha=OFFSCALE_MARKER_ALPHA,
            zorder=5,
            clip_on=False,
        )

    for xi, yi, n in zip(x[in_range], means[in_range], panel.loc[in_range, "n"], strict=True):
        if int(n) < 5:
            ax.text(xi, yi, f"n={int(n)}", fontsize=5.8, color=color, ha="center", va="bottom")
    return int(np.count_nonzero(offscale)), float(np.nanmax(means[valid]))


def offscale_marker_y(ax: plt.Axes, upper: float) -> float:
    lower = ax.get_ylim()[0]
    if ax.get_yscale() == "log":
        return float(np.exp(np.log(lower) + 0.982 * (np.log(upper) - np.log(lower))))
    return lower + 0.985 * (upper - lower)


def contiguous_segments(indices: np.ndarray) -> list[np.ndarray]:
    if indices.size == 0:
        return []
    breaks = np.where(np.diff(indices) > 1)[0] + 1
    return [segment for segment in np.split(indices, breaks) if segment.size]


def save_figure(
    fig: plt.Figure,
    *,
    outdir: Path,
    stem: str,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = outdir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi if fmt == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def caption_text(metric: MetricSpec, missingness: str, report: FigureReport) -> str:
    caption = (
        f"{metric.label} for {MISSINGNESS_LABELS[missingness]} missingness. "
        "Rows are benchmark problems and columns are inference strategies. "
        "The x-axis shows low, mid, and high simulation budgets, displayed as "
        "approximately 10^3, 10^4, and 10^5 simulations. Missing-data methods show "
        "10%, 25%, and 50% missing fractions; Full-Data NPE is shown as a neutral "
        "baseline. Points are means across five independently trained estimators. "
        "Error bars show mean +/- 1.96 standard errors across training seeds. "
        "Reference metrics first summarize reference observations within each seed "
        "using the established seed-level CSV summaries."
    )
    if metric.yscale == "log":
        caption += " The y-axis is logarithmic; the dashed horizontal reference line denotes a ratio of 1."
    if report.offscale_points:
        caption += (
            " Values exceeding the displayed y-axis range are indicated by upward-facing "
            "boundary markers and omitted from the visible range for readability."
        )
    caption += (
        f" Display range: [{format_limit(report.display_range[0])}, "
        f"{format_limit(report.display_range[1])}]. Off-scale aggregate points: "
        f"{report.offscale_points}. Maximum plotted aggregate value before display limiting: "
        f"{format_limit(report.max_actual)}."
    )
    return caption


def combined_caption_text(metric: MetricSpec, report: FigureReport) -> str:
    caption = (
        f"{metric.label} comparing MCAR and MNAR missingness. Rows are benchmark "
        "problems and columns are inference strategies. The x-axis shows low, mid, "
        "and high simulation budgets, displayed as approximately 10^3, 10^4, and "
        "10^5 simulations. Missing-data methods show 10%, 25%, and 50% missing "
        "fractions; color encodes missing fraction. MCAR curves are solid, while "
        "MNAR curves are dashed. Full-Data NPE is shown once as a neutral baseline "
        "because missingness does not apply. Points are means across five "
        "independently trained estimators. Error bars show mean +/- 1.96 standard "
        "errors across training seeds. Reference metrics first summarize reference "
        "observations within each seed using the established seed-level CSV summaries."
    )
    if metric.yscale == "log":
        caption += " The y-axis is logarithmic; the dashed horizontal reference line denotes a ratio of 1."
    if report.offscale_points:
        caption += (
            " Values exceeding the displayed y-axis range are indicated by upward-facing "
            "boundary markers; MNAR off-scale markers are open triangles."
        )
    caption += (
        f" Display range: [{format_limit(report.display_range[0])}, "
        f"{format_limit(report.display_range[1])}]. Off-scale aggregate points: "
        f"{report.offscale_points}. Maximum plotted aggregate value before display limiting: "
        f"{format_limit(report.max_actual)}."
    )
    return caption


def format_limit(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    if abs(value) >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def print_figure_report(report: FigureReport) -> None:
    lo, hi = report.display_range
    missingness_label = (
        "MCAR+MNAR"
        if report.missingness == "mcar_mnar"
        else MISSINGNESS_LABELS[report.missingness]
    )
    print(f"{report.metric_key} / {missingness_label}:")
    print(f"  display range: [{format_limit(lo)}, {format_limit(hi)}]")
    print(f"  y-scale: {report.yscale}")
    print(f"  off-scale points: {report.offscale_points}")
    print(f"  maximum actual aggregate value: {format_limit(report.max_actual)}")


def write_captions(captions: dict[str, str], outdir: Path) -> Path:
    path = outdir / "captions.md"
    merged = read_existing_captions(path)
    merged.update(captions)
    text = ["# Benchmark Figure Captions", ""]
    for stem, caption in sorted(merged.items()):
        text.append(f"## `{stem}`")
        text.append("")
        text.append(textwrap.fill(caption, width=100))
        text.append("")
    path.write_text("\n".join(text), encoding="utf-8")
    return path


def read_existing_captions(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    captions: dict[str, str] = {}
    current_stem: str | None = None
    current_lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## `") and line.endswith("`"):
            if current_stem is not None:
                captions[current_stem] = " ".join(current_lines).strip()
            current_stem = line.removeprefix("## `").removesuffix("`")
            current_lines = []
        elif current_stem is not None:
            if line.strip():
                current_lines.append(line.strip())
    if current_stem is not None:
        captions[current_stem] = " ".join(current_lines).strip()
    return captions


def selected_jobs(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.all:
        return [(metric, missingness) for metric in METRICS for missingness in MISSINGNESS_ORDER]
    if not args.metric or not args.missingness:
        raise SystemExit("Pass --all or both --metric and --missingness.")
    return [(args.metric, args.missingness)]


def selected_combined_jobs(args: argparse.Namespace) -> list[str]:
    if args.all:
        return list(METRICS)
    if not args.metric:
        raise SystemExit("Pass --all or --metric with --combined-mcar-mnar.")
    return [args.metric]


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


def main() -> None:
    args = parse_args()
    formats = args.formats or ["pdf", "png"]
    configure_matplotlib()

    df = load_results(args.input)
    validate_results(df)
    df = successful_rows(df)

    written: list[Path] = []
    captions: dict[str, str] = {}
    reports: list[FigureReport] = []
    if args.combined_mcar_mnar:
        for metric_key in selected_combined_jobs(args):
            metric = METRICS[metric_key]
            stem = f"{metric.filename_prefix}_mcar_mnar"
            fig, report = plot_combined_missingness_grid(df, metric=metric)
            reports.append(report)
            print_figure_report(report)
            written.extend(save_figure(fig, outdir=args.outdir, stem=stem, formats=formats, dpi=args.dpi))
            captions[stem] = combined_caption_text(metric, report)
    else:
        for metric_key, missingness in selected_jobs(args):
            metric = METRICS[metric_key]
            stem = f"{metric.filename_prefix}_{missingness}"
            fig, report = plot_benchmark_grid(df, metric=metric, missingness=missingness)
            reports.append(report)
            print_figure_report(report)
            written.extend(save_figure(fig, outdir=args.outdir, stem=stem, formats=formats, dpi=args.dpi))
            captions[stem] = caption_text(metric, missingness, report)

    caption_path = write_captions(captions, args.outdir)
    print("Wrote figures:")
    for path in written:
        print(f"  {path}")
    print(f"Wrote captions: {caption_path}")


if __name__ == "__main__":
    main()
