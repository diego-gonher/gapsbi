"""Generate quality-gated GLU/MNAR partial-data reference posteriors by IS."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import h5py
import numpy as np
from gapsbi.masks import MeanNormalizedSelfCensoringMNARMask
from gapsbi.references import (DEFAULT_PARTIAL_DATA_REFERENCE_VERSION, load_reference_posteriors_hdf5,
    sample_glu_mnar_missing_importance, sample_glu_reference_posterior,
    save_glu_mcar_partial_data_reference_posteriors_hdf5)

def main():
 p=argparse.ArgumentParser(); p.add_argument('--missing-fraction',type=float,required=True); p.add_argument('--proposals',type=int,default=1_000_000); p.add_argument('--samples',type=int,default=10_000); p.add_argument('--seed',type=int,default=456789); p.add_argument('--reference-mask-seed',type=int,default=987654); p.add_argument('--overwrite',action='store_true'); a=p.parse_args()
 full_obs,_,full_meta,_=load_reference_posteriors_hdf5('references/reference_posteriors_v1/glu_references.h5')
 dataset=Path(f'data/canonical_v1/glu/mnar/glu_mnar_self_censoring_mean_normalized_identity_eps{round(a.missing_fraction*100):03d}_seed123.h5')
 with h5py.File(dataset) as h: mask_meta=json.loads(h.attrs['mask'])
 gen=MeanNormalizedSelfCensoringMNARMask(missing_fraction=float(mask_meta['missing_fraction']),eps=float(mask_meta['eps']),score_transform=str(mask_meta['score_transform']))
 mask=gen.generate(full_obs['x_full'],full_obs['theta_true'],np.random.default_rng(a.reference_mask_seed)); xobs=full_obs['x_full']*mask
 rng=np.random.default_rng(a.seed); theta=np.empty((10,a.samples,10)); ess=[]; essfrac=[]; maxw=[]; runtime=[]; validation=[]
 for i in range(10):
  t=time.perf_counter(); missing=np.flatnonzero(mask[i]==0)
  if len(missing):
   latent,d=sample_glu_mnar_missing_importance(x_obs=xobs[i],mask=mask[i],prior_low=np.full(10,-1.),prior_high=np.full(10,1.),simulator_scale=.1,missing_fraction=a.missing_fraction,num_proposals=a.proposals,num_samples=a.samples,rng=rng)
   if d['ess']<20_000 or d['ess_fraction']<.02 or d['max_weight']>.001: raise RuntimeError(f'observation {i} failed IS quality gate: {d}')
   complete=np.repeat(xobs[i][None],a.samples,axis=0); complete[:,missing]=latent; ess.append(d['ess']);essfrac.append(d['ess_fraction']);maxw.append(d['max_weight'])
  else:
   complete=np.repeat(xobs[i][None],a.samples,axis=0); ess.append(float(a.proposals));essfrac.append(1.);maxw.append(1/a.proposals)
  theta[i]=sample_glu_reference_posterior(x_full=complete,prior_low=np.full(10,-1.),prior_high=np.full(10,1.),simulator_scale=.1,num_reference_samples=1,rng=rng)[:,0]; runtime.append(time.perf_counter()-t)
 # Independent IS check on the most-missing observation, without retaining a second artifact.
 j=int(np.argmax((mask==0).sum(1))); validation=[j]
 if np.any(mask[j]==0):
  z,_=sample_glu_mnar_missing_importance(x_obs=xobs[j],mask=mask[j],prior_low=np.full(10,-1.),prior_high=np.full(10,1.),simulator_scale=.1,missing_fraction=a.missing_fraction,num_proposals=a.proposals,num_samples=a.samples,rng=np.random.default_rng(a.seed+10_000))
  c=np.repeat(xobs[j][None],a.samples,axis=0);c[:,mask[j]==0]=z; vtheta=sample_glu_reference_posterior(x_full=c,prior_low=np.full(10,-1.),prior_high=np.full(10,1.),simulator_scale=.1,num_reference_samples=1,rng=np.random.default_rng(a.seed+10_001))[:,0]
  validation_mean_delta=float(np.abs(vtheta.mean(0)-theta[j].mean(0)).max())
 else: validation_mean_delta=0.
 obs={'theta_true':full_obs['theta_true'],'x_full':full_obs['x_full'],'x_obs':xobs,'mask':mask}
 meta={'problem':'glu','missingness_mechanism':'mnar','posterior_method':'prior_predictive_importance_sampling_then_exact_truncated_normal','posterior_conditioning':'theta_given_x_obs_mask','mask':mask_meta,'simulator':full_meta['simulator'],'prior_low':[-1.]*10,'prior_high':[1.]*10,'simulator_scale':.1,'reference_mask_seed':a.reference_mask_seed,'proposal_seed':a.seed,'num_proposals':a.proposals,'resampling':'numpy_choice_weighted_with_replacement','quality_gate_ess':20000,'quality_gate_ess_fraction':.02,'quality_gate_max_weight':.001,'reference_observation_source':'references/reference_posteriors_v1/glu_references.h5'}
 diag={'importance_ess':np.array(ess),'importance_ess_fraction':np.array(essfrac),'importance_max_normalized_weight':np.array(maxw),'importance_num_proposals':np.full(10,a.proposals),'runtime_sec':np.array(runtime),'independent_validation_observation_indices':np.array(validation),'independent_validation_theta_mean_abs_max_delta':np.array([validation_mean_delta])}
 out=Path('references')/DEFAULT_PARTIAL_DATA_REFERENCE_VERSION/f'glu_mnar_eps{round(a.missing_fraction*100):03d}_references.h5'
 save_glu_mcar_partial_data_reference_posteriors_hdf5(out,observations=obs,theta_samples=theta,metadata=meta,diagnostics=diag,reference_version=DEFAULT_PARTIAL_DATA_REFERENCE_VERSION,overwrite=a.overwrite)
 print(json.dumps({'path':str(out),'min_ess':min(ess),'min_frac':min(essfrac),'max_weight':max(maxw),'runtime':sum(runtime),'validation_delta':validation_mean_delta}))
if __name__=='__main__': main()
