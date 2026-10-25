"""Evaluate LV/MNAR parameter-level importance weights from saved proposal chains."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from gapsbi.masks import LotkaVolterraLogTotalMNARMask
from gapsbi.references import (importance_reweight_lotka_volterra_mnar, load_reference_posteriors_hdf5)
from gapsbi.simulators import LotkaVolterraSimulator

CASES=((.10,0),(.25,5),(.50,1))
ROOT=Path('outputs/diagnostics/lv_mnar_is_pilot')
OUT=ROOT/'importance_sampling_results.json'

def sim(meta):
 s=meta['simulator']; return LotkaVolterraSimulator(num_timepoints=s['num_timepoints'],days=s['days'],observation_noise_scale=s['observation_noise_scale'],initial_state=tuple(s['initial_state']),prior_log_mean=np.array(s['prior_log_mean']),prior_log_std=np.array(s['prior_log_std']),max_state=s['max_state'],ode_rtol=s['ode_rtol'],ode_atol=s['ode_atol'])
def systematic(w,rng):
 u=(rng.random()+np.arange(len(w)))/len(w); return np.searchsorted(np.cumsum(w),u)
def main():
 obs,_,meta,_=load_reference_posteriors_hdf5('references/reference_posteriors_v1/lotka_volterra_references.h5'); simulator=sim(meta); out={'schema':'gapsbi_lv_mnar_importance_pilot_v1','cases':{}}
 for fraction,index in CASES:
  tag=f'eps{round(fraction*100):03d}_ref{index:02d}'; p=ROOT/f'lv_mcar_{tag}_pilot.json'; record=json.loads(p.read_text())
  generator=LotkaVolterraLogTotalMNARMask(fraction); mask=generator.generate(obs['x_full'],obs['theta_true'],np.random.default_rng(987654))[index]; xobs=obs['x_full'][index]*mask
  case={'mechanism':'mnar','proposal_target':'p(theta|x_obs_mnar)','runs':[],'mask_metadata':generator.metadata()}
  for ensemble in range(2):
   chain=np.load(ROOT/f'lv_mcar_{tag}_ensemble{ensemble:02d}_chains.npz')['log_theta_chain']; tau=np.asarray(record['ensembles'][ensemble]['autocorr_time_heuristic']); proposals=chain[::max(1,int(np.ceil(tau.max())))].reshape(-1,4)
   for k in (1,4,16):
    for replica in range(2):
     started=time.perf_counter(); theta,d=importance_reweight_lotka_volterra_mnar(proposals,xobs,mask,simulator=simulator,missing_fraction=fraction,latent_completions=k,rng=np.random.default_rng([4242,round(100*fraction),index,ensemble,k,replica]))
     w=d.pop('parameter_weights'); res=systematic(w,np.random.default_rng([888,round(100*fraction),index,ensemble,k,replica])); sample=theta[res]; top=max(1,int(np.ceil(.01*len(w))));
     case['runs'].append({'ensemble':ensemble,'latent_completions':k,'replica':replica,'runtime_seconds':time.perf_counter()-started,'num_theta_proposals':len(theta),'theta_autocorr_stride':int(np.ceil(tau.max())),**d,'top_1_percent_weight_mass':float(np.sort(w)[-top:].sum()),'unique_theta_after_systematic_resampling':int(len(np.unique(res))),'posterior_mean':np.average(theta,axis=0,weights=w),'posterior_covariance':np.cov(theta,rowvar=False,aweights=w),'posterior_quantiles_05_50_95':np.quantile(sample,[.05,.5,.95],axis=0)})
  out['cases'][tag]=case; OUT.write_text(json.dumps(out,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x) if isinstance(x,np.floating) else int(x)),encoding='utf8')
 print(OUT)
if __name__=='__main__': main()
