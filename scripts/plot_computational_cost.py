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
DEFAULT_OUTPUT_DIR = Path("outputs/analysis/figures/computational_cost")
DEFAULT_AGGREGATE_PATH = Path("outputs/analysis/aggregates/aggregate_computational_cost.csv")

# Runtime semantics, traced to the experiment train.py scripts:
# - training_time_sec times only inference.train(...), i.e. estimator training.
# - reference_posterior_sampling_time_sec times only sample_posteriors_once(...)
#   on the reference observations. It excludes inverse transforms, HDF5 writing,
#   C2ST/mean-shift/covariance metrics, TARP/SBC diagnostics, and plotting.
#   We divide it by num_reference_eval to report sampling time per benchmark
#   reference observation for num_reference_posterior_samples posterior draws.
# - posterior_sampling_time_sec times the separate synthetic test/calibration
#   posterior block used for TARP/SBC arrays, so it is not used as the primary
#   inference-cost quantity here.
# - diagnostics_time_sec, reference_metrics_time_sec, and total_runtime_sec
#   include diagnostic/evaluation work and are intentionally not plotted.
TRAINING_COLUMN = "training_time_sec"
REFERENCE_SAMPLING_COLUMN = "reference_posterior_sampling_time_sec"
REFERENCE_COUNT_COLUMN = "num_reference_eval"
REFERENCE_NUM_SAMPLES_COLUMN = "num_reference_posterior_samples"
INFERENCE_COLUMN = "inference_time_per_observation_sec"

BUDGET_ORDER = ["low_sim_budget", "mid_sim_budget", "high_sim_budget"]
BUDGET_CLI = {
    "low": "low_sim_budget",
    "mid": "mid_sim_budget",
    "high": "high_sim_budget",
}
BUDGET_LABELS = {
    "low_sim_budget": "low simulation budget",
    "mid_sim_budget": "mid simulation budget",
    "high_sim_budget": "high simulation budget",
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
    "npe_full_data": "Full-Data\nNPE",
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
GROUP_WIDTH = 0.72
BAR_WIDTH = GROUP_WIDTH / 3.0


@dataclass(frozen=True)
class CostSpec:
    key: str
    column: str
    title: str
    mean_column: str
    std_column: str
    se_column: str


COST_SPECS = {
    "training": CostSpec(
        key="training",
        column=TRAINING_COLUMN,
        title="Training",
        mean_column="training_time_mean",
        std_column="training_time_std",
        se_column="training_time_se",
    ),
    "inference": CostSpec(
        key="inference",
        column=INFERENCE_COLUMN,
        title="Inference / sampling",
        mean_column="inference_time_mean",
        std_column="inference_time_std",
        se_column="inference_time_se",
    ),
}


@dataclass(frozen=True)
class AxisScale:
    unit_label: str
    divisor: float
    log: bool
    ylim: tuple[float, float]
    raw_min: float
    raw_max: float


@dataclass(frozen=True)
class FigureReport:
    stem: str
    budget: str
    missingness: str
    training_scale: AxisScale
    inference_scale: AxisScale


@dataclass(frozen=True)
class TrainingOnlyReport:
    stem: str
    budget: str
    missingness: str
    training_scale: AxisScale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GapSBI computational cost figures.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--aggregate-out", type=Path, default=DEFAULT_AGGREGATE_PATH)
    parser.add_argument("--budget", choices=sorted(BUDGET_CLI), help="Budget to plot.")
    parser.add_argument("--missingness", choices=MISSINGNESS_ORDER, help="Missingness to plot.")
    parser.add_argument("--all", action="store_true", help="Generate all budget/missingness figures.")
    parser.add_argument(
        "--training-only",
        action="store_true",
        help="Generate single-column training-cost figures only.",
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


def normalize_epsilon(value: Any) -> float:
    if pd.isna(value):
        return math.nan
    epsilon = float(value)
    return epsilon / 100.0 if epsilon > 1.0 else epsilon


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = df.copy()
    out["epsilon"] = out["epsilon"].map(normalize_epsilon).round(2)
    for col in ["budget", "method", "problem", "missingness"]:
        out[col] = out[col].astype(str)
    out["seed"] = out["seed"].astype(str)

    required = [TRAINING_COLUMN, REFERENCE_SAMPLING_COLUMN, REFERENCE_COUNT_COLUMN]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"Missing required timing columns: {missing}")

    out[TRAINING_COLUMN] = pd.to_numeric(out[TRAINING_COLUMN], errors="coerce")
    out[REFERENCE_SAMPLING_COLUMN] = pd.to_numeric(out[REFERENCE_SAMPLING_COLUMN], errors="coerce")
    out[REFERENCE_COUNT_COLUMN] = pd.to_numeric(out[REFERENCE_COUNT_COLUMN], errors="coerce")
    out[INFERENCE_COLUMN] = out[REFERENCE_SAMPLING_COLUMN] / out[REFERENCE_COUNT_COLUMN]
    return out


def successful_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "compile_status" in out:
        out = out[out["compile_status"].astype(str).str.lower() == "ok"]
    if "diagnostics_status" in out:
        out = out[out["diagnostics_status"].astype(str).str.lower() == "ok"]
    return out


def validate_results(df: pd.DataFrame) -> None:
    timing_cols = [col for col in df.columns if "time" in col.lower() or "runtime" in col.lower()]
    print("Computational-cost validation")
    print(f"Timing columns found: {', '.join(timing_cols)}")
    print("Timing definitions:")
    print("  training_time_sec: elapsed wall-clock time for inference.train(...) only.")
    print(
        "  reference_posterior_sampling_time_sec: elapsed wall-clock time for "
        "sample_posteriors_once(...) on reference observations only."
    )
    print(
        "  inference_time_per_observation_sec: reference_posterior_sampling_time_sec / "
        "num_reference_eval; excludes metric evaluation, saving, diagnostics, and plotting."
    )
    print("  posterior_sampling_time_sec: synthetic test/calibration posterior block; not plotted.")
    print("  reference_metrics_time_sec/diagnostics_time_sec/total_runtime_sec: not plotted.")

    methods = set(df["method"].unique())
    problems = set(df["problem"].unique())
    budgets = set(df["budget"].unique())
    missingness = set(df["missingness"].unique())
    fractions = sorted(
        round(float(eps), 2)
        for eps in df.loc[df["method"] != "npe_full_data", "epsilon"].dropna().unique()
    )
    print(f"Methods found: {', '.join(sorted(methods))}")
    print(f"Problems found: {', '.join(PROBLEM_LABELS[p] for p in PROBLEM_ORDER)}")
    print(f"Budgets found: {', '.join(BUDGET_LABELS[b] for b in BUDGET_ORDER)}")
    print(f"Missingness mechanisms found: {', '.join(sorted(missingness))}")
    print(f"Missing fractions found: {', '.join(FRACTION_LABELS[f] for f in FRACTION_ORDER)}")

    missing_methods = set(METHOD_ORDER) - methods
    missing_problems = set(PROBLEM_ORDER) - problems
    missing_budgets = set(BUDGET_ORDER) - budgets
    missing_mechanisms = (set(MISSINGNESS_ORDER) | {"none"}) - missingness
    if missing_methods or missing_problems or missing_budgets or missing_mechanisms:
        raise ValueError(
            "Missing expected metadata: "
            f"methods={sorted(missing_methods)}, problems={sorted(missing_problems)}, "
            f"budgets={sorted(missing_budgets)}, missingness={sorted(missing_mechanisms)}"
        )
    if fractions != FRACTION_ORDER:
        raise ValueError(f"Unexpected missing fractions {fractions}; expected {FRACTION_ORDER}.")

    counts = seed_counts(df)
    print("Seed counts per configuration:")
    print(counts["num_valid_timing_seeds"].value_counts().sort_index().to_string())
    incomplete = counts[counts["num_valid_timing_seeds"] < 5]
    print(f"Configurations with fewer than 5 valid timing measurements: {len(incomplete)}")
    if len(incomplete):
        print(incomplete.to_string(index=False))

    for key, spec in COST_SPECS.items():
        values = pd.to_numeric(df[spec.column], errors="coerce")
        finite = values[np.isfinite(values)]
        print(
            f"Runtime min/max by cost type: {key} "
            f"{finite.min():.4g} to {finite.max():.4g} seconds"
        )


def seed_counts(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[
        df[TRAINING_COLUMN].notna()
        & df[INFERENCE_COLUMN].notna()
        & np.isfinite(df[TRAINING_COLUMN])
        & np.isfinite(df[INFERENCE_COLUMN])
    ].copy()
    return (
        valid.groupby(["budget", "problem", "missingness", "method", "epsilon"], dropna=False)
        .agg(num_valid_timing_seeds=("seed", "nunique"))
        .reset_index()
    )


def build_cost_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[
        df[TRAINING_COLUMN].notna()
        & df[INFERENCE_COLUMN].notna()
        & np.isfinite(df[TRAINING_COLUMN])
        & np.isfinite(df[INFERENCE_COLUMN])
    ].copy()
    grouped = (
        valid.groupby(["budget", "problem", "missingness", "method", "epsilon"], dropna=False)
        .agg(
            num_seeds=("seed", "nunique"),
            training_time_mean=(TRAINING_COLUMN, "mean"),
            training_time_std=(TRAINING_COLUMN, "std"),
            inference_time_mean=(INFERENCE_COLUMN, "mean"),
            inference_time_std=(INFERENCE_COLUMN, "std"),
            reference_posterior_sampling_time_total_mean=(REFERENCE_SAMPLING_COLUMN, "mean"),
            num_reference_eval=(REFERENCE_COUNT_COLUMN, "median"),
            num_reference_posterior_samples=(REFERENCE_NUM_SAMPLES_COLUMN, "median"),
        )
        .reset_index()
    )
    grouped["training_time_se"] = grouped["training_time_std"] / np.sqrt(grouped["num_seeds"])
    grouped["inference_time_se"] = grouped["inference_time_std"] / np.sqrt(grouped["num_seeds"])
    grouped.loc[grouped["num_seeds"] <= 1, ["training_time_se", "inference_time_se"]] = 0.0
    ordered_cols = [
        "budget",
        "problem",
        "missingness",
        "method",
        "epsilon",
        "num_seeds",
        "training_time_mean",
        "training_time_std",
        "training_time_se",
        "inference_time_mean",
        "inference_time_std",
        "inference_time_se",
        "reference_posterior_sampling_time_total_mean",
        "num_reference_eval",
        "num_reference_posterior_samples",
    ]
    return grouped[ordered_cols]


def write_aggregate(agg: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(path, index=False)
    return path


def selected_jobs(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.all:
        return [(budget, missingness) for budget in BUDGET_ORDER for missingness in MISSINGNESS_ORDER]
    if not args.budget or not args.missingness:
        raise SystemExit("Pass --all or both --budget and --missingness.")
    return [(BUDGET_CLI[args.budget], args.missingness)]


def rows_for_figure(agg: pd.DataFrame, budget: str, missingness: str) -> pd.DataFrame:
    full = agg[(agg["budget"] == budget) & (agg["method"] == "npe_full_data")]
    missing = agg[
        (agg["budget"] == budget)
        & (agg["missingness"] == missingness)
        & (agg["method"] != "npe_full_data")
    ]
    return pd.concat([full, missing], ignore_index=True)


def choose_axis_scale(values_sec: np.ndarray) -> AxisScale:
    finite = values_sec[np.isfinite(values_sec) & (values_sec > 0)]
    if finite.size == 0:
        return AxisScale("s", 1.0, False, (0.0, 1.0), math.nan, math.nan)

    raw_min = float(np.min(finite))
    raw_max = float(np.max(finite))
    if raw_max >= 3600:
        unit_label, divisor = "h", 3600.0
    elif raw_max >= 180:
        unit_label, divisor = "min", 60.0
    else:
        unit_label, divisor = "s", 1.0

    scaled = finite / divisor
    ratio = raw_max / raw_min if raw_min > 0 else math.inf
    use_log = ratio > 10.0
    if use_log:
        ymin = max(float(np.min(scaled)) * 0.7, 1e-4)
        ymax = float(np.max(scaled)) * 1.35
    else:
        ymin = 0.0
        ymax = float(np.max(scaled)) * 1.25
    return AxisScale(unit_label, divisor, use_log, (ymin, ymax), raw_min, raw_max)


def scales_for_figure(rows: pd.DataFrame) -> dict[str, AxisScale]:
    return {
        key: choose_axis_scale(pd.to_numeric(rows[spec.mean_column], errors="coerce").to_numpy(dtype=float))
        for key, spec in COST_SPECS.items()
    }


def cost_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=FULL_DATA_COLOR, marker="s", linewidth=0, markersize=7, label="Full-data NPE"),
        *[
            Line2D(
                [0],
                [0],
                color=FRACTION_COLORS[epsilon],
                marker="s",
                linewidth=0,
                markersize=7,
                label=FRACTION_LABELS[epsilon],
            )
            for epsilon in FRACTION_ORDER
        ],
    ]


def plot_cost_figure(agg: pd.DataFrame, *, budget: str, missingness: str) -> tuple[plt.Figure, FigureReport]:
    rows = rows_for_figure(agg, budget, missingness)
    scales = scales_for_figure(rows)
    fig, axes = plt.subplots(
        nrows=len(PROBLEM_ORDER),
        ncols=2,
        figsize=(11.0, 8.2),
        sharex=True,
        constrained_layout=False,
    )

    for row_idx, problem in enumerate(PROBLEM_ORDER):
        for col_idx, key in enumerate(["training", "inference"]):
            spec = COST_SPECS[key]
            ax = axes[row_idx, col_idx]
            style_axis(ax, scales[key])
            if row_idx == 0:
                ax.set_title(spec.title, fontsize=10.5, fontweight="bold", pad=8)
            if col_idx == 0:
                ax.set_ylabel(PROBLEM_LABELS[problem], fontsize=10.5, fontweight="bold")
            plot_cost_bars(ax, rows, problem=problem, spec=spec, scale=scales[key])

            if row_idx < len(PROBLEM_ORDER) - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xticks(np.arange(len(METHOD_ORDER)))
                ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], fontsize=8)
            if col_idx == 1:
                ax.tick_params(labelleft=True)

    title = f"Computational cost under {MISSINGNESS_LABELS[missingness]} missingness - {BUDGET_LABELS[budget]}"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.988)
    fig.supxlabel("Inference strategy", fontsize=11, y=0.040)

    axes[0, 0].text(
        -0.16,
        1.12,
        f"Wall-clock time [{scales['training'].unit_label}]",
        transform=axes[0, 0].transAxes,
        fontsize=10,
        ha="left",
        va="center",
    )
    axes[0, 1].text(
        -0.08,
        1.12,
        f"Wall-clock time [{scales['inference'].unit_label}]",
        transform=axes[0, 1].transAxes,
        fontsize=10,
        ha="left",
        va="center",
    )

    fig.legend(
        handles=cost_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
        fontsize=9,
        columnspacing=1.5,
        handletextpad=0.4,
    )
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.12, top=0.835, wspace=0.18, hspace=0.18)
    report = FigureReport(
        stem=f"cost_{budget.split('_')[0]}_{missingness}",
        budget=budget,
        missingness=missingness,
        training_scale=scales["training"],
        inference_scale=scales["inference"],
    )
    return fig, report


def plot_training_only_figure(
    agg: pd.DataFrame,
    *,
    budget: str,
    missingness: str,
) -> tuple[plt.Figure, TrainingOnlyReport]:
    rows = rows_for_figure(agg, budget, missingness)
    spec = COST_SPECS["training"]
    scale = choose_axis_scale(
        pd.to_numeric(rows[spec.mean_column], errors="coerce").to_numpy(dtype=float)
    )
    fig, axes = plt.subplots(
        nrows=len(PROBLEM_ORDER),
        ncols=1,
        figsize=(5.8, 8.2),
        sharex=True,
        constrained_layout=False,
    )

    for row_idx, problem in enumerate(PROBLEM_ORDER):
        ax = axes[row_idx]
        style_axis(ax, scale)
        ax.set_ylabel(PROBLEM_LABELS[problem], fontsize=10.5, fontweight="bold")
        plot_cost_bars(ax, rows, problem=problem, spec=spec, scale=scale)
        if row_idx < len(PROBLEM_ORDER) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xticks(np.arange(len(METHOD_ORDER)))
            ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], fontsize=8)

    fig.suptitle("Training Cost", fontsize=13, fontweight="bold", y=0.988)
    fig.supxlabel("Inference strategy", fontsize=11, y=0.040)
    fig.supylabel(f"Wall-clock time [{scale.unit_label}]", fontsize=11, x=0.018)
    fig.legend(
        handles=cost_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
        fontsize=9,
        columnspacing=1.15,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.22, right=0.985, bottom=0.12, top=0.845, hspace=0.18)
    report = TrainingOnlyReport(
        stem=f"training_cost_{budget.split('_')[0]}_{missingness}",
        budget=budget,
        missingness=missingness,
        training_scale=scale,
    )
    return fig, report


def style_axis(ax: plt.Axes, scale: AxisScale) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.55)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b8b8b8")
    ax.spines["bottom"].set_color("#b8b8b8")
    ax.tick_params(axis="both", labelsize=8, length=2.5, color="#777777")
    if scale.log:
        ax.set_yscale("log")
    ax.set_ylim(*scale.ylim)
    ax.set_axisbelow(True)


def plot_cost_bars(ax: plt.Axes, rows: pd.DataFrame, *, problem: str, spec: CostSpec, scale: AxisScale) -> None:
    for method_idx, method in enumerate(METHOD_ORDER):
        if method == "npe_full_data":
            row = rows[(rows["problem"] == problem) & (rows["method"] == method)]
            if row.empty:
                continue
            draw_bar(ax, method_idx, row.iloc[0], spec, scale, FULL_DATA_COLOR, GROUP_WIDTH)
            continue

        for offset, epsilon in zip([-BAR_WIDTH, 0.0, BAR_WIDTH], FRACTION_ORDER, strict=True):
            row = rows[
                (rows["problem"] == problem)
                & (rows["method"] == method)
                & np.isclose(rows["epsilon"].astype(float), epsilon)
            ]
            if row.empty:
                continue
            draw_bar(
                ax,
                method_idx + offset,
                row.iloc[0],
                spec,
                scale,
                FRACTION_COLORS[epsilon],
                BAR_WIDTH * 0.98,
            )


def draw_bar(
    ax: plt.Axes,
    x: float,
    row: pd.Series,
    spec: CostSpec,
    scale: AxisScale,
    color: str,
    width: float,
) -> None:
    mean = float(row[spec.mean_column]) / scale.divisor
    se = float(row[spec.se_column]) / scale.divisor
    if not np.isfinite(mean):
        return
    ax.bar(x, mean, width=width, color=color, edgecolor="none", alpha=0.95, zorder=2)
    ax.errorbar(
        [x],
        [mean],
        yerr=[1.96 * se if np.isfinite(se) else 0.0],
        color="#333333",
        linewidth=0.55,
        elinewidth=0.6,
        capsize=1.6,
        capthick=0.6,
        zorder=3,
    )
    if int(row["num_seeds"]) < 5:
        ax.text(x, mean, f"n={int(row['num_seeds'])}", fontsize=6, ha="center", va="bottom")


def save_figure(fig: plt.Figure, *, outdir: Path, stem: str, formats: list[str], dpi: int) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = outdir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi if fmt == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def caption_text(report: FigureReport) -> str:
    ref_samples = "the configured number of reference posterior samples"
    return (
        f"Computational cost for {MISSINGNESS_LABELS[report.missingness]} missingness at the "
        f"{BUDGET_LABELS[report.budget]}. Rows are benchmark problems and columns show training "
        "and inference/sampling cost. Bars are means across five independent training seeds; "
        "error bars show mean +/- 1.96 standard errors across seeds. Training time is the "
        "wall-clock duration of inference.train(...) only. Inference/sampling time is "
        "reference_posterior_sampling_time_sec divided by num_reference_eval, i.e. the "
        f"wall-clock time to draw {ref_samples} for one benchmark reference observation after "
        "training. This excludes reference metric computation, TARP/SBC diagnostics, plotting, "
        "saving, and total experiment overhead. Full-Data NPE is shown once because missing "
        "fraction does not apply."
    )


def training_only_caption_text(report: TrainingOnlyReport) -> str:
    return (
        f"Training cost for {MISSINGNESS_LABELS[report.missingness]} missingness at the "
        f"{BUDGET_LABELS[report.budget]}. Rows are benchmark problems and bars show "
        "Full-Data NPE plus the five missing-data representation strategies. Missing-data "
        "methods show 10%, 25%, and 50% missing fractions; Full-Data NPE is shown once "
        "because missing fraction does not apply. Bars are means across five independent "
        "training seeds; error bars show mean +/- 1.96 standard errors across seeds. "
        "Training time is the wall-clock duration of inference.train(...) or the "
        "corresponding joint estimator-training routine only; metric computation, "
        "posterior sampling, diagnostics, plotting, saving, and total pipeline overhead "
        "are excluded."
    )


def write_captions(captions: dict[str, str], outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "captions.md"
    merged = read_existing_captions(path)
    merged.update(captions)
    lines = ["# Computational Cost Figure Captions", ""]
    for stem, caption in sorted(merged.items()):
        lines.append(f"## `{stem}`")
        lines.append("")
        lines.append(textwrap.fill(caption, width=100))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
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
        elif current_stem is not None and line.strip():
            current_lines.append(line.strip())
    if current_stem is not None:
        captions[current_stem] = " ".join(current_lines).strip()
    return captions


def print_report(report: FigureReport) -> None:
    print(f"{report.stem}:")
    print(
        "  training range: "
        f"{report.training_scale.raw_min:.4g} to {report.training_scale.raw_max:.4g} s; "
        f"unit={report.training_scale.unit_label}; log={report.training_scale.log}"
    )
    print(
        "  inference range: "
        f"{report.inference_scale.raw_min:.4g} to {report.inference_scale.raw_max:.4g} s/obs; "
        f"unit={report.inference_scale.unit_label}; log={report.inference_scale.log}"
    )


def print_training_only_report(report: TrainingOnlyReport) -> None:
    print(f"{report.stem}:")
    print(
        "  training range: "
        f"{report.training_scale.raw_min:.4g} to {report.training_scale.raw_max:.4g} s; "
        f"unit={report.training_scale.unit_label}; log={report.training_scale.log}"
    )


def print_budget_ranges(agg: pd.DataFrame) -> None:
    for budget in BUDGET_ORDER:
        rows = agg[agg["budget"] == budget]
        train = rows["training_time_mean"].dropna()
        infer = rows["inference_time_mean"].dropna()
        print(
            f"{budget}: mean training-time range "
            f"{train.min():.4g} to {train.max():.4g} s; "
            f"mean inference-time range {infer.min():.4g} to {infer.max():.4g} s/obs"
        )


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
    agg = build_cost_aggregate(df)
    aggregate_path = write_aggregate(agg, args.aggregate_out)

    written: list[Path] = []
    captions: dict[str, str] = {}
    for budget, missingness in selected_jobs(args):
        if args.training_only:
            fig, report = plot_training_only_figure(agg, budget=budget, missingness=missingness)
            print_training_only_report(report)
            written.extend(save_figure(fig, outdir=args.outdir, stem=report.stem, formats=formats, dpi=args.dpi))
            captions[report.stem] = training_only_caption_text(report)
        else:
            fig, report = plot_cost_figure(agg, budget=budget, missingness=missingness)
            print_report(report)
            written.extend(save_figure(fig, outdir=args.outdir, stem=report.stem, formats=formats, dpi=args.dpi))
            captions[report.stem] = caption_text(report)

    caption_path = write_captions(captions, args.outdir)
    print_budget_ranges(agg)

    missing_timing = seed_counts(df)
    missing_timing = missing_timing[missing_timing["num_valid_timing_seeds"] < 5]
    print(f"Configurations with missing timing information: {len(missing_timing)}")
    if len(missing_timing):
        print(missing_timing.to_string(index=False))

    print(f"Wrote aggregate CSV: {aggregate_path}")
    print("Wrote figures:")
    for path in written:
        print(f"  {path}")
    print(f"Wrote captions: {caption_path}")


if __name__ == "__main__":
    main()
