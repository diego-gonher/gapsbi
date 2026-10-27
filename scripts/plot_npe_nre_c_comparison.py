"""Plot existing seed aggregates as a reusable 3x6 NPE/NRE-C figure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


METRIC_ORDER = ("c2st", "mean_shift", "covariance_trace_ratio")
METRIC_LABELS = {"c2st": "C2ST accuracy", "mean_shift": "Posterior mean shift", "covariance_trace_ratio": "Covariance-trace ratio"}
METHOD_ORDER = ("full_data", "zero_imputation", "mask_augmentation", "learned_constant_imputation", "probabilistic_learned_imputation", "masked_transformer_embedding")
METHOD_LABELS = {"full_data":"Full-data baseline", "zero_imputation":"Zero Imputation", "mask_augmentation":"Zero + Mask", "learned_constant_imputation":"Learned Constant", "probabilistic_learned_imputation":"Probabilistic Learned Imputation", "masked_transformer_embedding":"Masked Transformer Embedding"}
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
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.2,"axes.linewidth":.7,"axes.labelcolor":"#222222","xtick.color":"#222222","ytick.color":"#222222","pdf.fonttype":42,"ps.fonttype":42,"figure.facecolor":"white","savefig.facecolor":"white"})


def plot_comparison(aggregate: pd.DataFrame, *, problem: str, mechanism: str, fraction: float, outdir: Path) -> list[Path]:
    frame=comparison_columns(aggregate)
    configure_style()
    fig,axes=plt.subplots(3,6,figsize=(19.2,8.0),sharex=True)
    for col,semantic in enumerate(METHOD_ORDER):
        for row,metric in enumerate(METRIC_ORDER):
            ax=axes[row,col]
            ax.grid(True,which="major",color="#D9D9D9",linewidth=.45,alpha=.8)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False)
            if metric=="c2st": ax.set_ylim(.5,1.0);ax.axhline(.5,color="#777777",linewidth=.55,linestyle=(0,(2,2)))
            elif metric=="mean_shift": ax.set_ylim(0,1.5)
            else: ax.set_yscale("log");ax.set_ylim(.8,1000);ax.axhline(1.,color="#777777",linewidth=.55,linestyle=(0,(2,2)))
            if row==0:
                title=METHOD_LABELS[semantic]
                if semantic=="full_data": title += "\n(full-reference target)"
                ax.set_title(title,fontsize=8.2,fontweight="bold",pad=8)
            if col==0: ax.set_ylabel(METRIC_LABELS[metric],fontsize=8.2)
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
    fig.suptitle(f"{problem.upper()} {mechanism.upper()} {fraction:.0%}: NPE and NRE-C reference metrics",fontsize=13,fontweight="bold",y=.995)
    fig.text(.5,.955,"Full-data baseline uses p(θ | x_full) and full-data references; missing-data methods use p(θ | x_obs, m) and the shared partial references. Interpret the baseline column separately.",ha="center",va="top",fontsize=8.2,color="#333333")
    handles=[Line2D([0],[0],color=COLORS[k],linestyle=STYLES[k][0],marker=STYLES[k][1],linewidth=1.2,markersize=3.2,label=k) for k in ("NPE","NRE-C")]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.923),ncol=2,frameon=False,fontsize=8.5)
    fig.supxlabel("Simulation budget (log scale)",y=.035,fontsize=9)
    fig.subplots_adjust(left=.065,right=.995,top=.865,bottom=.09,wspace=.28,hspace=.22)
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
