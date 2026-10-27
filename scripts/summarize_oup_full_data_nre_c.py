"""Summarize experimental OUP full-data NRE-C against saved NPE results."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

SEEDS=(101,202,303,404,505)
METRICS=("c2st_accuracy","posterior_mean_shift","covariance_trace_ratio")

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('outputs/diagnostics/nre_c_oup_full_data/comparison.json'));a=p.parse_args(); rows=[]
 for budget in ('low','mid','high'):
  nre=[]
  for seed in SEEDS:
   d=json.load(open(Path(f'outputs_{budget}_sim_budget/nre_c_full_data/oup_full_data/seed_{seed}/reference_evaluation.json')))
   t=json.load(open(Path(f'outputs_{budget}_sim_budget/nre_c_full_data/oup_full_data/seed_{seed}/training_summary.json')))
   nre.append({'seed':seed,**d['reference_metric_aggregate'],'training_runtime_sec':t['training_runtime_sec'],'reference_inference_runtime_sec':d.get('inference_runtime_sec')})
  npe=json.load(open(Path(f'outputs_{budget}_sim_budget/npe_full_data/oup_full_data/all_results.json')))
  for method,items in (('nre_c',nre),('npe',[{'seed':x['seed'],'c2st_accuracy':float(np.mean(x['reference_c2st_accuracy'])),'posterior_mean_shift':float(np.median(x['reference_posterior_mean_shift'])),'covariance_trace_ratio':float(np.median(x['reference_covariance_trace_ratio'])),'training_runtime_sec':x['training_time_sec'],'reference_inference_runtime_sec':x.get('reference_posterior_sampling_time_sec')} for x in npe])):
   rows.extend({'budget':budget,'method':method,**item} for item in items)
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(rows,indent=2)+'\n')
 fig,axes=plt.subplots(1,3,figsize=(10,3)); labels=['C2ST accuracy','Mean shift','Covariance trace ratio']
 for ax,metric,label in zip(axes,METRICS,labels):
  for method,marker in (('npe','o'),('nre_c','s')):
   values=[np.array([r[metric] for r in rows if r['budget']==b and r['method']==method]) for b in ('low','mid','high')]
   ax.errorbar(range(3),[v.mean() for v in values],yerr=[v.std(ddof=1) for v in values],marker=marker,label=method.upper())
  ax.set_xticks(range(3),['low','mid','high']);ax.set_title(label)
 axes[0].legend();fig.tight_layout();fig.savefig(a.output.with_suffix('.png'),dpi=200);plt.close(fig)
if __name__=='__main__':main()
