"""Pilot comparison of GLU/MNAR latent-x samplers; writes no reference artifact."""
from __future__ import annotations
import json, time
from pathlib import Path
import arviz as az
import h5py
import numpy as np
from gapsbi.references import (
    sample_glu_mnar_missing_emcee, sample_glu_mnar_missing_importance,
    sample_glu_mnar_missing_independence_mh, sample_glu_reference_posterior,
)

OUT=Path("outputs/analysis/glu_mnar_importance_pilot.json")

def theta_from_latent(xobs, mask, latent, rng):
    full=np.repeat(xobs[None,:],len(latent),axis=0); full[:,mask==0]=latent
    return sample_glu_reference_posterior(x_full=full,prior_low=np.full(10,-1.),prior_high=np.full(10,1.),simulator_scale=.1,num_reference_samples=1,rng=rng)[:,0]

def moments(a):
    return {"mean":a.mean(0).tolist(),"variance":a.var(0).tolist(),"q05":np.quantile(a,.05,axis=0).tolist(),"q50":np.quantile(a,.5,axis=0).tolist(),"q95":np.quantile(a,.95,axis=0).tolist()}

def main():
    result={"settings":{"is_proposals":[100000,500000,1000000],"num_draws":10000,"emcee_ensembles":4,"emcee_walkers":32,"emcee_burnin":1000,"emcee_steps":5000}}
    for eps in (.1,.25,.5):
        p=Path(f"outputs_low_sim_budget/npe_mask_augmentation/glu/mnar/glu_mnar_eps{int(eps*100):03d}/seed_101/reference_posterior_samples.h5")
        with h5py.File(p) as f: x,m=f["x_obs"][...],f["mask"][...]
        idx=int(np.argmax((m==0).sum(1))); xo,mo=x[idx],m[idx]; kw=dict(x_obs=xo,mask=mo,prior_low=np.full(10,-1.),prior_high=np.full(10,1.),simulator_scale=.1,missing_fraction=eps)
        chains=[]; emcee_lat=[]; emcee_times=[]; diag=[]
        for seed in range(4):
            t=time.perf_counter(); z,d=sample_glu_mnar_missing_emcee(**kw,num_samples=10000,rng=np.random.default_rng(1000+seed),nwalkers=32,burnin=1000,steps=5000); emcee_times.append(time.perf_counter()-t); emcee_lat.append(z); diag.append(d)
            # One trace from each independently seeded ensemble; walkers are not chains.
            chains.append(d["ensemble_mean_trace"])
        em=np.concatenate(emcee_lat); em_theta=theta_from_latent(xo,mo,em,np.random.default_rng(90))
        idata=az.from_dict(posterior={"x_missing":np.asarray(chains)})
        emdiag={"runtime_sec_each":emcee_times,"acceptance_fraction":[d["acceptance_fraction"] for d in diag],"autocorr_time":[np.asarray(d["autocorr_time"]).tolist() for d in diag],"rhat":az.rhat(idata).x_missing.values.tolist(),"ess_bulk":az.ess(idata,method="bulk").x_missing.values.tolist(),"ess_tail":az.ess(idata,method="tail").x_missing.values.tolist()}
        t=time.perf_counter(); mh,mhd=sample_glu_mnar_missing_independence_mh(**kw,num_samples=10000,rng=np.random.default_rng(44),burnin=10000,thin=10); mh_time=time.perf_counter()-t; mh_theta=theta_from_latent(xo,mo,mh,np.random.default_rng(91))
        importance=[]
        for n in (100000,500000,1000000):
            t=time.perf_counter(); z,d=sample_glu_mnar_missing_importance(**kw,num_proposals=n,num_samples=10000,rng=np.random.default_rng(n)); importance.append({"proposals":n,"runtime_sec":time.perf_counter()-t,"diagnostics":d,"latent":moments(z),"theta":moments(theta_from_latent(xo,mo,z,np.random.default_rng(n+1)))})
        result[str(eps)]={"reference_index":idx,"num_missing":int((mo==0).sum()),"emcee":{"diagnostics":emdiag,"latent":moments(em),"theta":moments(em_theta)},"mh":{"runtime_sec":mh_time,"diagnostics":mhd,"latent":moments(mh),"theta":moments(mh_theta)},"importance":importance}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(result,indent=2)); print(OUT)
if __name__=="__main__": main()
