"""Plot existing seed aggregates as a reusable 3x6 NPE/NRE-C figure."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# Support both direct script execution and loading by path in tests/notebooks.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_benchmark_summary import METRICS as BENCHMARK_METRICS
from plot_benchmark_summary import configure_matplotlib, style_axis


METRIC_ORDER = ("c2st", "mean_shift", "covariance_trace_ratio")
METRIC_LABELS = {"c2st": "C2ST\naccuracy", "mean_shift": "Posterior\nmean shift", "covariance_trace_ratio": "Covariance\ntrace ratio"}
METHOD_ORDER = ("full_data", "zero_imputation", "mask_augmentation", "learned_constant_imputation", "probabilistic_learned_imputation", "masked_transformer_embedding")
METHOD_LABELS = {"full_data":"Full-Data Baseline\n(full-data refs)", "zero_imputation":"Zero", "mask_augmentation":"Zero +\nMask", "learned_constant_imputation":"Learned\nConstant", "probabilistic_learned_imputation":"Probabilistic", "masked_transformer_embedding":"Masked\nTransformer"}
METHOD_NAMES = {"full_data":("npe_full_data","nre_c_full_data"), "zero_imputation":("npe_zero_imputation","nre_c_zero_imputation"), "mask_augmentation":("npe_mask_augmentation","nre_c_mask_augmentation"), "learned_constant_imputation":("npe_learned_constant_imputation","nre_c_learned_constant_imputation"), "probabilistic_learned_imputation":("npe_probabilistic_learned_imputation","nre_c_probabilistic_learned_imputation"), "masked_transformer_embedding":("npe_masked_transformer_embedding","nre_c_masked_transformer_embedding")}
BUDGETS = ("low", "mid", "high")
BUDGET_X = np.asarray([1e3, 1e4, 1e5])
COLORS = {"NPE":"#2878B5", "NRE-C":"#E07A1F"}
STYLES = {"NPE":("-", "o"), "NRE-C":((0,(4,2.5)), "s")}


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=Path("outputs/analysis/nre_c_comparison/oup_mcar_eps010/comparison_aggregate_metrics.csv"))
    p.add_argument("--outdir",type=Path,default=None)
    p.add_argument("--problem",default=None)
    p.add_argument("--mechanism",default=None)
    p.add_argument("--fraction",type=float,default=None)
    return p.parse_args()


def comparison_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a long aggregate frame and reject absent/duplicate panel values."""
    required={"backbone","method","budget","target","metric","mean","std","ci95_low","ci95_high","n_seeds"}
    missing=required-set(frame.columns)
    if missing: raise ValueError(f"Aggregate input lacks fields: {sorted(missing)}")
    out=frame.copy()
    out["budget"]=out["budget"].astype(str)
    if out.duplicated(["backbone","method","budget","target","metric"]).any():
        raise ValueError("Aggregate input contains duplicate backbone/method/budget/target/metric rows.")
    return out


def panel_series(frame: pd.DataFrame, metric: str, semantic_method: str, backbone: str) -> pd.DataFrame:
    method = METHOD_NAMES[semantic_method][0 if backbone=="NPE" else 1]
    target="full" if semantic_method=="full_data" else "partial"
    subset=frame[(frame.metric==metric)&(frame.method==method)&(frame.backbone==backbone)&(frame.target==target)].set_index("budget").reindex(BUDGETS)
    return subset


def configure_style() -> None:
    # Keep typography and exported-font behavior in sync with the main benchmark.
    configure_matplotlib()


def plot_comparison(aggregate: pd.DataFrame, *, problem: str, mechanism: str, fraction: float, outdir: Path) -> list[Path]:
    frame=comparison_columns(aggregate)
    configure_style()
    # The benchmark uses 13.2 x 8.4 inches for four rows. Preserve each panel's
    # proportions by scaling the comparison height to three rows.
    fig,axes=plt.subplots(3,6,figsize=(13.2,6.3),sharex=True)
    for col,semantic in enumerate(METHOD_ORDER):
        for row,metric in enumerate(METRIC_ORDER):
            ax=axes[row,col]
            spec=BENCHMARK_METRICS[metric]
            style_axis(ax,spec,spec.ylim)
            if row==0:
                title=METHOD_LABELS[semantic]
                ax.set_title(title,fontsize=10.5,fontweight="bold",pad=8)
            if col==0: ax.set_ylabel(METRIC_LABELS[metric],fontsize=11.5,fontweight="bold")
            for backbone in ("NPE","NRE-C"):
                data=panel_series(frame,metric,semantic,backbone)
                if data["mean"].isna().all(): continue
                means=data["mean"].to_numpy(dtype=float)
                lower=np.maximum(means-data["ci95_low"].to_numpy(dtype=float),0)
                upper=np.maximum(data["ci95_high"].to_numpy(dtype=float)-means,0)
                if metric=="covariance_trace_ratio":
                    lower=np.minimum(lower,np.maximum(means-.8,0))
                line,marker=STYLES[backbone]
                ax.errorbar(BUDGET_X,means,yerr=np.vstack([lower,upper]),color=COLORS[backbone],linestyle=line,marker=marker,markersize=3.1,linewidth=1.15,capsize=1.7,elinewidth=.65,markerfacecolor=COLORS[backbone],label=backbone)
            ax.set_xscale("log");ax.set_xlim(7e2,1.4e5);ax.set_xticks(BUDGET_X,[r"$10^3$",r"$10^4$",r"$10^5$"])
            if row<2: ax.tick_params(labelbottom=False)
            else: ax.tick_params(axis="x",labelsize=9.8)
            if col>0: ax.tick_params(labelleft=False)
    fig.suptitle(f"{problem.upper()} {mechanism.upper()} {fraction:.0%}: NPE and NRE-C",fontsize=14,fontweight="bold",y=.988)
    handles=[Line2D([0],[0],color=COLORS[k],linestyle=STYLES[k][0],marker=STYLES[k][1],linewidth=1.2,markersize=3.2,label=k) for k in ("NPE","NRE-C")]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.955),ncol=2,frameon=False,fontsize=10,handlelength=2.0,columnspacing=1.5)
    fig.supxlabel("Simulation budget",y=.040,fontsize=12)
    fig.subplots_adjust(left=.075,right=.995,bottom=.10,top=.825,wspace=.10,hspace=.16)
    outdir.mkdir(parents=True,exist_ok=True)
    paths=[]
    stem=f"{problem}_{mechanism}_eps{round(fraction*100):03d}_npe_vs_nre_c_3x6"
    for extension in ("pdf","png"):
        path=outdir/f"{stem}.{extension}";fig.savefig(path,dpi=300 if extension=="png" else None,bbox_inches="tight");paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    a=parse_args();frame=pd.read_csv(a.input)
    meta_path=a.input.parent/"provenance.json"
    meta=json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    problem=a.problem or meta.get("problem","unknown");mechanism=a.mechanism or meta.get("mechanism","unknown");fraction=a.fraction if a.fraction is not None else float(meta.get("fraction",.1))
    outdir=a.outdir or a.input.parent/"figures"
    paths=plot_comparison(frame,problem=problem,mechanism=mechanism,fraction=fraction,outdir=outdir)
    print("\n".join(map(str,paths)))


if __name__=="__main__": main()
