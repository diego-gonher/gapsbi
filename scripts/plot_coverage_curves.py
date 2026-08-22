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
DEFAULT_OUTPUT_DIR = Path("outputs/analysis/figures/coverage")

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
IDEAL_COLOR = "#6e6e6e"


@dataclass(frozen=True)
class CoverageCurve:
    alpha: np.ndarray
    mean: np.ndarray
    ci95: np.ndarray
    num_seeds: int
    num_tarp_eval: int | None
    num_tarp_posterior_samples: int | None


@dataclass(frozen=True)
class ValidationReport:
    missing_configs: list[str]
    alpha_grid: np.ndarray
    num_tarp_eval: set[int]
    num_tarp_posterior_samples: set[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mean TARP empirical coverage curves.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--budget", choices=sorted(BUDGET_CLI), help="Budget to plot.")
    parser.add_argument("--missingness", choices=MISSINGNESS_ORDER, help="Missingness to plot.")
    parser.add_argument("--all", action="store_true", help="Generate all budget/missingness figures.")
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
    for col in ["budget", "method", "problem", "missingness", "seed"]:
        out[col] = out[col].astype(str)
    if "compile_status" in out:
        out = out[out["compile_status"].astype(str).str.lower() == "ok"]
    if "diagnostics_status" in out:
        out = out[out["diagnostics_status"].astype(str).str.lower() == "ok"]
    return out


def diagnostics_path(row: pd.Series) -> Path:
    for col in ("diagnostics_arrays_resolved_path", "diagnostics_arrays_path"):
        value = str(row.get(col, "")).strip()
        if value and value.lower() != "nan":
            path = Path(value)
            return path if path.is_absolute() else Path.cwd() / path
    run_dir = str(row.get("run_dir", "")).strip()
    if not run_dir or run_dir.lower() == "nan":
        raise ValueError("row has no diagnostics path or run_dir")
    path = Path(run_dir) / "diagnostics_arrays.npz"
    return path if path.is_absolute() else Path.cwd() / path


def load_seed_curve(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    path = diagnostics_path(row)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as arrays:
        if "alpha" not in arrays or "ecp" not in arrays:
            raise ValueError(f"{path} is missing alpha/ecp arrays")
        alpha = np.asarray(arrays["alpha"], dtype=float).reshape(-1)
        ecp = np.asarray(arrays["ecp"], dtype=float).reshape(-1)
    if alpha.shape != ecp.shape:
        raise ValueError(f"{path} alpha/ecp shape mismatch: {alpha.shape} != {ecp.shape}")
    if alpha.size == 0:
        raise ValueError(f"{path} alpha/ecp arrays are empty")
    if np.nanmin(alpha) < -1e-8 or np.nanmax(alpha) > 1 + 1e-8:
        raise ValueError(f"{path} alpha grid outside [0, 1]")
    if np.nanmin(ecp) < -1e-8 or np.nanmax(ecp) > 1 + 1e-8:
        raise ValueError(f"{path} ecp values outside [0, 1]")
    return alpha, ecp


def validation_report(df: pd.DataFrame) -> ValidationReport:
    required_columns = {"diagnostics_arrays_path", "num_tarp_eval", "num_tarp_posterior_samples"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required coverage columns: {sorted(missing_columns)}")

    expected_methods = set(METHOD_ORDER)
    expected_problems = set(PROBLEM_ORDER)
    expected_budgets = set(BUDGET_ORDER)
    expected_missingness = set(MISSINGNESS_ORDER) | {"none"}
    methods = set(df["method"].unique())
    problems = set(df["problem"].unique())
    budgets = set(df["budget"].unique())
    missingness = set(df["missingness"].unique())
    missing = {
        "methods": sorted(expected_methods - methods),
        "problems": sorted(expected_problems - problems),
        "budgets": sorted(expected_budgets - budgets),
        "missingness": sorted(expected_missingness - missingness),
    }
    if any(missing.values()):
        raise ValueError(f"Missing expected metadata: {missing}")

    missing_methods_df = df[df["method"] != "npe_full_data"]
    fractions = sorted(
        {
            round(float(value), 2)
            for value in missing_methods_df["epsilon"].dropna().unique()
        }
    )
    if fractions != FRACTION_ORDER:
        raise ValueError(f"Unexpected normalized missing fractions: {fractions}")

    missing_configs: list[str] = []
    config_rows: list[dict[str, Any]] = []
    alpha_grid: np.ndarray | None = None
    for budget in BUDGET_ORDER:
        for problem in PROBLEM_ORDER:
            rows = df[
                (df["budget"] == budget)
                & (df["problem"] == problem)
                & (df["method"] == "npe_full_data")
                & (df["missingness"] == "none")
            ]
            config_rows.append(config_summary("full", budget, problem, "none", None, rows))
            alpha_grid = update_alpha_grid(alpha_grid, rows, missing_configs)

            for missingness_value in MISSINGNESS_ORDER:
                for method in METHOD_ORDER[1:]:
                    for epsilon in FRACTION_ORDER:
                        rows = select_rows(
                            df,
                            budget=budget,
                            problem=problem,
                            method=method,
                            missingness=missingness_value,
                            epsilon=epsilon,
                        )
                        config_rows.append(
                            config_summary(method, budget, problem, missingness_value, epsilon, rows)
                        )
                        alpha_grid = update_alpha_grid(alpha_grid, rows, missing_configs)

    incomplete = [row for row in config_rows if row["num_seeds"] != 5]
    for row in incomplete:
        missing_configs.append(
            f"{row['budget']}/{row['problem']}/{row['method']}/"
            f"{row['missingness']}/eps={row['epsilon']}: seeds={row['num_seeds']}"
        )

    if alpha_grid is None:
        raise ValueError("No coverage alpha/ecp arrays were loaded.")

    num_tarp_eval = {int(v) for v in pd.to_numeric(df["num_tarp_eval"], errors="coerce").dropna().unique()}
    num_tarp_posterior_samples = {
        int(v)
        for v in pd.to_numeric(df["num_tarp_posterior_samples"], errors="coerce").dropna().unique()
    }

    seed_counts = pd.DataFrame(config_rows)["num_seeds"].value_counts().sort_index()
    print("Coverage curve validation")
    print("Coverage field/file discovered: diagnostics_arrays.npz arrays `alpha` and `ecp`")
    print(
        "Coverage diagnostic definition: TARP empirical coverage probability; "
        "`ecp[a] = mean(tarp_prob <= a)` over evaluation observations, where "
        "tarp_prob is the fraction of posterior samples closer to a random reference point "
        "than the true theta."
    )
    print(
        "Coverage source code: src/gapsbi/evaluation/tarp.py::compute_tarp_from_samples; "
        "scalar TARP MAE is computed later from abs(ecp - alpha)."
    )
    print(f"Nominal credibility grid: {alpha_grid.size} points from {alpha_grid[0]:.2f} to {alpha_grid[-1]:.2f}")
    print(f"Number of evaluation observations: {sorted(num_tarp_eval)}")
    print(f"Posterior samples per evaluation observation: {sorted(num_tarp_posterior_samples)}")
    print(f"Methods found: {', '.join(sorted(methods))}")
    print(f"Problems found: {', '.join(PROBLEM_LABELS[p] for p in PROBLEM_ORDER)}")
    print(f"Budgets found: {', '.join(BUDGET_LABELS[b] for b in BUDGET_ORDER)}")
    print(f"Missingness mechanisms found: {', '.join(sorted(missingness))}")
    print(f"Normalized missing fractions found: {', '.join(FRACTION_LABELS[f] for f in FRACTION_ORDER)}")
    print("Seed counts per configuration:")
    print(seed_counts.to_string())
    print(f"Configurations missing coverage data or fewer than 5 seeds: {len(missing_configs)}")
    for item in missing_configs:
        print(f"  {item}")
    return ValidationReport(
        missing_configs=missing_configs,
        alpha_grid=alpha_grid,
        num_tarp_eval=num_tarp_eval,
        num_tarp_posterior_samples=num_tarp_posterior_samples,
    )


def config_summary(
    method: str,
    budget: str,
    problem: str,
    missingness: str,
    epsilon: float | None,
    rows: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "method": method,
        "budget": budget,
        "problem": problem,
        "missingness": missingness,
        "epsilon": epsilon,
        "num_seeds": int(rows["seed"].nunique()),
        "num_rows": int(len(rows)),
    }


def update_alpha_grid(
    alpha_grid: np.ndarray | None,
    rows: pd.DataFrame,
    missing_configs: list[str],
) -> np.ndarray | None:
    for _, row in rows.iterrows():
        try:
            alpha, _ = load_seed_curve(row)
        except Exception as exc:
            missing_configs.append(f"{row.get('summary_path', '')}: {exc}")
            continue
        if alpha_grid is None:
            alpha_grid = alpha
        elif alpha.shape != alpha_grid.shape or not np.allclose(alpha, alpha_grid, atol=1e-8, rtol=0):
            raise ValueError("Coverage alpha grids are inconsistent across seeds/configurations.")
    return alpha_grid


def select_rows(
    df: pd.DataFrame,
    *,
    budget: str,
    problem: str,
    method: str,
    missingness: str,
    epsilon: float | None,
) -> pd.DataFrame:
    rows = df[
        (df["budget"] == budget)
        & (df["problem"] == problem)
        & (df["method"] == method)
    ]
    if method == "npe_full_data":
        return rows[rows["missingness"] == "none"]
    if epsilon is None:
        raise ValueError("epsilon is required for missing-data methods")
    return rows[
        (rows["missingness"] == missingness)
        & np.isclose(rows["epsilon"].astype(float), epsilon)
    ]


def prepare_curve(rows: pd.DataFrame) -> CoverageCurve | None:
    alphas = []
    ecps = []
    for _, row in rows.sort_values("seed").iterrows():
        alpha, ecp = load_seed_curve(row)
        alphas.append(alpha)
        ecps.append(ecp)
    if not ecps:
        return None
    alpha0 = alphas[0]
    if any(alpha.shape != alpha0.shape or not np.allclose(alpha, alpha0, atol=1e-8, rtol=0) for alpha in alphas):
        raise ValueError("Coverage alpha grids differ within configuration")
    values = np.vstack(ecps)
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1) if values.shape[0] > 1 else np.zeros(values.shape[1])
    ci95 = 1.96 * std / math.sqrt(values.shape[0]) if values.shape[0] > 1 else np.zeros(values.shape[1])
    num_tarp_eval = unique_int(rows["num_tarp_eval"])
    num_tarp_posterior_samples = unique_int(rows["num_tarp_posterior_samples"])
    return CoverageCurve(
        alpha=alpha0,
        mean=mean,
        ci95=ci95,
        num_seeds=values.shape[0],
        num_tarp_eval=num_tarp_eval,
        num_tarp_posterior_samples=num_tarp_posterior_samples,
    )


def unique_int(series: pd.Series) -> int | None:
    values = pd.to_numeric(series, errors="coerce").dropna().astype(int).unique()
    if len(values) == 1:
        return int(values[0])
    return None


def plot_coverage_grid(
    df: pd.DataFrame,
    *,
    budget: str,
    missingness: str,
) -> plt.Figure:
    fig, axes = plt.subplots(
        nrows=len(PROBLEM_ORDER),
        ncols=len(METHOD_ORDER),
        figsize=(13.2, 9.1),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )

    for row_idx, problem in enumerate(PROBLEM_ORDER):
        for col_idx, method in enumerate(METHOD_ORDER):
            ax = axes[row_idx, col_idx]
            style_axis(ax)
            if row_idx == 0:
                ax.set_title(METHOD_LABELS[method], fontsize=9.5, fontweight="bold", pad=8)
            if col_idx == 0:
                ax.set_ylabel(PROBLEM_LABELS[problem], fontsize=10.5, fontweight="bold")

            ax.plot(
                [0.0, 1.0],
                [0.0, 1.0],
                color=IDEAL_COLOR,
                linestyle=(0, (3, 3)),
                linewidth=0.75,
                alpha=0.65,
                zorder=1,
            )

            if method == "npe_full_data":
                rows = select_rows(
                    df,
                    budget=budget,
                    problem=problem,
                    method=method,
                    missingness="none",
                    epsilon=None,
                )
                curve = prepare_curve(rows)
                if curve is not None:
                    draw_curve(ax, curve, color=FULL_DATA_COLOR)
            else:
                for epsilon in FRACTION_ORDER:
                    rows = select_rows(
                        df,
                        budget=budget,
                        problem=problem,
                        method=method,
                        missingness=missingness,
                        epsilon=epsilon,
                    )
                    curve = prepare_curve(rows)
                    if curve is not None:
                        draw_curve(ax, curve, color=FRACTION_COLORS[epsilon])

            if row_idx < len(PROBLEM_ORDER) - 1:
                ax.tick_params(labelbottom=False)
            if col_idx > 0:
                ax.tick_params(labelleft=False)

    title = (
        f"Mean posterior coverage under {MISSINGNESS_LABELS[missingness]} missingness - "
        f"{BUDGET_LABELS[budget]}"
    )
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.988)
    fig.supxlabel("Nominal credibility level", fontsize=11, y=0.035)
    fig.supylabel("Empirical coverage", fontsize=11, x=0.012)
    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=5,
        frameon=False,
        fontsize=9,
        handlelength=2.1,
        columnspacing=1.4,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.09, top=0.825, wspace=0.12, hspace=0.20)
    return fig


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(1)
    ticks = np.linspace(0.0, 1.0, 6)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid(axis="both", color="#e9e9e9", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b8b8b8")
    ax.spines["bottom"].set_color("#b8b8b8")
    ax.tick_params(axis="both", labelsize=7.8, length=2.4, color="#777777")


def draw_curve(ax: plt.Axes, curve: CoverageCurve, *, color: str) -> None:
    lower = np.clip(curve.mean - curve.ci95, 0.0, 1.0)
    upper = np.clip(curve.mean + curve.ci95, 0.0, 1.0)
    ax.fill_between(curve.alpha, lower, upper, color=color, alpha=0.11, linewidth=0, zorder=2)
    ax.plot(curve.alpha, curve.mean, color=color, linewidth=1.25, alpha=0.96, zorder=3)


def legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=FULL_DATA_COLOR, linewidth=1.35, label="Full-data NPE"),
        *[
            Line2D([0], [0], color=FRACTION_COLORS[epsilon], linewidth=1.35, label=FRACTION_LABELS[epsilon])
            for epsilon in FRACTION_ORDER
        ],
        Line2D([0], [0], color=IDEAL_COLOR, linewidth=0.85, linestyle=(0, (3, 3)), label="Ideal coverage"),
    ]


def selected_jobs(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.all:
        return [(budget, missingness) for budget in BUDGET_ORDER for missingness in MISSINGNESS_ORDER]
    if not args.budget or not args.missingness:
        raise SystemExit("Pass --all or both --budget and --missingness.")
    return [(BUDGET_CLI[args.budget], args.missingness)]


def save_figure(fig: plt.Figure, *, outdir: Path, stem: str, dpi: int) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = [outdir / f"{stem}.pdf", outdir / f"{stem}.png"]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return paths


def caption_text(budget: str, missingness: str, report: ValidationReport) -> str:
    num_eval = sorted(report.num_tarp_eval)
    num_samples = sorted(report.num_tarp_posterior_samples)
    return (
        f"Mean TARP empirical posterior coverage curves for {MISSINGNESS_LABELS[missingness]} "
        f"missingness at the {BUDGET_LABELS[budget]}. Rows show OUP, GLM, GLU, and "
        "Lotka-Volterra; columns show Full-Data NPE, Zero, Zero + Mask, Learned Constant, "
        "Probabilistic, and Masked Transformer. Missing-data methods show 10%, 25%, and 50% "
        "missing fractions; Full-Data NPE is shown once in black because missing fraction does "
        "not apply. Curves show empirical posterior coverage as a function of nominal "
        "credibility level, averaged pointwise over five independently trained estimators. "
        "Shaded bands indicate mean +/- 1.96 standard errors across training seeds. The dashed "
        "diagonal denotes ideal calibration, empirical coverage equals nominal credibility. "
        f"Each seed-level curve was computed from {format_values(num_eval)} evaluation "
        f"observations and {format_values(num_samples)} posterior samples per observation using "
        "the saved TARP diagnostics arrays."
    )


def format_values(values: list[int]) -> str:
    if len(values) == 1:
        return f"{values[0]:,}"
    return ", ".join(f"{value:,}" for value in values)


def write_captions(captions: dict[str, str], outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "captions.md"
    text = ["# Coverage Curve Captions", ""]
    for stem, caption in sorted(captions.items()):
        text.append(f"## `{stem}`")
        text.append("")
        text.append(textwrap.fill(caption, width=100))
        text.append("")
    path.write_text("\n".join(text), encoding="utf-8")
    return path


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


def print_posthoc_summary(df: pd.DataFrame, jobs: list[tuple[str, str]]) -> None:
    print("Coverage curve pathology summary")
    for budget, missingness in jobs:
        rows = []
        for problem in PROBLEM_ORDER:
            for method in METHOD_ORDER:
                epsilons = [None] if method == "npe_full_data" else FRACTION_ORDER
                for epsilon in epsilons:
                    selected = select_rows(
                        df,
                        budget=budget,
                        problem=problem,
                        method=method,
                        missingness=missingness if method != "npe_full_data" else "none",
                        epsilon=epsilon,
                    )
                    curve = prepare_curve(selected)
                    if curve is None:
                        continue
                    mae = float(np.mean(np.abs(curve.mean - curve.alpha)))
                    max_abs = float(np.max(np.abs(curve.mean - curve.alpha)))
                    rows.append((mae, max_abs, problem, method, epsilon))
        if rows:
            worst = max(rows, key=lambda item: item[0])
            print(
                f"  {budget}/{missingness}: worst mean abs coverage error="
                f"{worst[0]:.4f}, max abs error={worst[1]:.4f} "
                f"({PROBLEM_LABELS[worst[2]]}, {METHOD_LABELS[worst[3]].replace(chr(10), ' ')}, "
                f"eps={worst[4]})"
            )


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    df = load_results(args.input)
    report = validation_report(df)
    jobs = selected_jobs(args)

    written: list[Path] = []
    captions: dict[str, str] = {}
    for budget, missingness in jobs:
        fig = plot_coverage_grid(df, budget=budget, missingness=missingness)
        stem = f"coverage_{budget.split('_', 1)[0]}_{missingness}"
        written.extend(save_figure(fig, outdir=args.outdir, stem=stem, dpi=args.dpi))
        captions[stem] = caption_text(budget, missingness, report)

    caption_path = write_captions(captions, args.outdir)
    print_posthoc_summary(df, jobs)
    print("Wrote coverage figures:")
    for path in written:
        print(f"  {path}")
    print(f"Wrote captions: {caption_path}")


if __name__ == "__main__":
    main()
