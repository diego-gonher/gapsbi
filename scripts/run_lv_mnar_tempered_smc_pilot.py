"""Resumable support-aware tempered SMC pilot for LV/MNAR reference 2."""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from gapsbi.masks import LotkaVolterraLogTotalMNARMask
from gapsbi.references import (load_reference_posteriors_hdf5,
    lotka_volterra_log_total_mnar_soft_log_probability)
from gapsbi.simulators import LotkaVolterraSimulator

ROOT = Path("outputs/diagnostics/lv_mnar_tempered_smc_pilot")

def enc(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    raise TypeError(type(x).__name__)
def dump(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2,default=enc));q.replace(p)
def savez(p,**a):
    p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp')
    with q.open('wb') as h: np.savez_compressed(h,**a)
    q.replace(p)
def sim(meta):
    s=meta['simulator'];return LotkaVolterraSimulator(num_timepoints=s['num_timepoints'],days=s['days'],observation_noise_scale=s['observation_noise_scale'],initial_state=tuple(s['initial_state']),prior_log_mean=np.asarray(s['prior_log_mean']),prior_log_std=np.asarray(s['prior_log_std']),max_state=s['max_state'],ode_rtol=s['ode_rtol'],ode_atol=s['ode_atol'])
def mu(z,S): return np.log(np.clip(S._solve_states(np.exp(z)).T.reshape(-1),1e-10,S.max_state))
def base(z,m,x,S):
    obs=np.asarray(m,dtype=bool); prior=np.sum(norm.logpdf(z,loc=S.prior_log_mean,scale=S.prior_log_std));return float(prior+np.sum(norm.logpdf(np.log(x[obs]),loc=m[obs],scale=S.observation_noise_scale)-np.log(x[obs])))
def ll(mu_,xi,x,m,S,f,d):
    y=np.asarray(x,float).copy();miss=np.asarray(m,dtype=bool)==0;y[miss]=np.exp(mu_[miss]+S.observation_noise_scale*xi)
    return lotka_volterra_log_total_mnar_soft_log_probability(y,m,missing_fraction=f,delta=d)
def target(base_,xi,ll_,beta): return base_-.5*(xi@xi)+beta*ll_
def normalize(logw):
    logw=np.asarray(logw);z=logsumexp(logw)
    if not np.isfinite(z): raise FloatingPointError('SMC particle collapse: all incremental weights zero.')
    w=np.exp(logw-z);return w,float(1/(w@w)),logw-z
def systematic(w,rng): return np.searchsorted(np.cumsum(w),(rng.random()+np.arange(len(w)))/len(w))
def choose_beta(beta, logl, logw, n, target_fraction=.70):
    if beta>=1:return 1.
    def ess_at(v): return normalize(logw+(v-beta)*logl)[1]
    if ess_at(1.)>=target_fraction*n:return 1.
    lo,hi=beta,1.
    for _ in range(32):
        mid=(lo+hi)/2
        if ess_at(mid)>=target_fraction*n:lo=mid
        else:hi=mid
    return lo
def ess_xi(xi,mu_,x,m,S,f,beta,d,rng,max_steps=1000):
    cur=beta*ll(mu_,xi,x,m,S,f,d)
    if not np.isfinite(cur): raise RuntimeError('infeasible particle passed to ESS')
    threshold=cur+np.log(rng.random());nu=rng.normal(size=xi.size);a=rng.uniform(0,2*np.pi);lo,hi=a-2*np.pi,a
    for k in range(1,max_steps+1):
        prop=xi*np.cos(a)+nu*np.sin(a);v=beta*ll(mu_,prop,x,m,S,f,d)
        if np.isfinite(v) and v>=threshold:return prop,k
        if a<0:lo=a
        else:hi=a
        a=rng.uniform(lo,hi)
    raise RuntimeError('ESS bracket exhausted')
def rejuvenate(z,xi,mus,bases,logl,x,m,S,f,beta,d,rng,steps):
    n=len(z);ode=ess_eval=hard=accept=0
    cov=np.cov(z,rowvar=False)+1e-8*np.eye(4);chol=np.linalg.cholesky(cov)
    miss=np.asarray(m,dtype=bool)==0
    for _ in range(steps):
      for i in range(n):
        xi[i],k=ess_xi(xi[i],mus[i],x,m,S,f,beta,d,rng);ess_eval+=k;logl[i]=ll(mus[i],xi[i],x,m,S,f,d)
        zp=z[i]+.65*(chol@rng.normal(size=4));ode+=1
        try: mp=mu(zp,S);xp=xi[i]+(mus[i][miss]-mp[miss])/S.observation_noise_scale;bp=base(zp,m,x,S);lp=ll(mp,xp,x,m,S,f,d)
        except Exception: hard+=1;continue
        old=target(bases[i],xi[i],logl[i],beta);new=target(bp,xp,lp,beta)
        if np.isfinite(new) and np.log(rng.random())<new-old:z[i],xi[i],mus[i],bases[i],logl[i]=zp,xp,mp,bp,lp;accept+=1
        elif not np.isfinite(new):hard+=1
    return ode,ess_eval,hard,accept
def checkpoint(root,stage,z,xi,mus,bases,logl,logw,state,diag,rng):
    savez(root/f'stage_{stage:03d}.npz',z=z,xi=xi,mu=mus,base=bases,logl=logl,logw=logw)
    dump(root/'progress.json',{**state,'stage':stage,'diagnostics':diag,'rng_state':rng.bit_generator.state})
def main():
 p=argparse.ArgumentParser();p.add_argument('--particles',type=int,default=512);p.add_argument('--run-index',type=int,required=True);p.add_argument('--seed',type=int,default=10331);p.add_argument('--rejuvenation-steps',type=int,default=2);a=p.parse_args()
 obs,_,meta,_=load_reference_posteriors_hdf5('references/reference_posteriors_v1/lotka_volterra_references.h5');S=sim(meta);f=.5;i=2;m=LotkaVolterraLogTotalMNARMask(f).generate(obs['x_full'],obs['theta_true'],np.random.default_rng(987654))[i];x=obs['x_full'][i]*m
 root=ROOT/f'eps050_ref02/run_{a.run_index:02d}';progress=root/'progress.json';rng=np.random.default_rng(np.random.SeedSequence([a.seed,a.run_index]))
 if progress.exists():
  P=json.loads(progress.read_text());last=np.load(root/f"stage_{P['stage']:03d}.npz");z,xi,mus,bases,logl,logw=[last[k] for k in ('z','xi','mu','base','logl','logw')];beta,delta,phase,index=P['beta'],P['delta'],P['phase'],P['index'];stage=P['stage']+1;diagnostics=P['diagnostics']
  rng.bit_generator.state=P['rng_state']
 else:
  source=Path('outputs/reference_generation/lv_mnar/eps050/reference_02');pieces=[]
  for e in range(2):
   chain=np.load(source/f'ensemble_{e}.npz')['log_theta_chain'];pieces.append(chain[::44].reshape(-1,4))
  pool=np.concatenate(pieces);z=pool[rng.choice(len(pool),a.particles,replace=False)];xi=rng.normal(size=(a.particles,int(np.sum(m==0))));mus=[];bases=[]
  for q in z: 
   v=mu(q,S);mus.append(v);bases.append(base(q,m,x,S))
  mus=np.asarray(mus);bases=np.asarray(bases);beta=0.;delta=.1;phase='beta';index=0;logl=np.array([ll(u,v,x,m,S,f,delta) for u,v in zip(mus,xi)]);logw=np.zeros(a.particles);stage=0;diagnostics=[]
 # beta is adaptively tempered at delta=.1, then delta follows a geometric softening schedule.
 path=[.03,.01,.003,.001,3e-4,1e-4,3e-5,1e-5,1e-6,1e-8,0.]
 while True:
  if phase=='beta':
   nb=choose_beta(beta,logl,logw,a.particles)
   inc=(nb-beta)*logl;beta=nb
   if beta>=1-1e-10:beta=1.;phase='delta'
  elif index<len(path):
   nd=path[index];new=np.array([ll(u,v,x,m,S,f,nd) for u,v in zip(mus,xi)])
   inc=beta*(new-logl);logl=new;delta=nd;index+=1
  else: break
  w,ess,logw=normalize(logw+inc);resampled=False
  # The adaptive beta solve deliberately targets 70% ESS.  Resample at that
  # same threshold; a lower threshold would leave beta unable to advance.
  if ess<=.70*a.particles + 1e-8:
   idx=systematic(w,rng);z,xi,mus,bases,logl=z[idx],xi[idx],mus[idx],bases[idx],logl[idx];logw=np.zeros(a.particles);w=np.full(a.particles,1/a.particles);resampled=True
  ode,ee,hard,acc=rejuvenate(z,xi,mus,bases,logl,x,m,S,f,beta,delta,rng,a.rejuvenation_steps)
  logl=np.array([ll(u,v,x,m,S,f,delta) for u,v in zip(mus,xi)])
  diag={'beta':beta,'delta':delta,'ess_before_resampling':ess,'resampled':resampled,'unique_particles_after_resampling':int(len(np.unique(idx))) if resampled else a.particles,'rejuvenation_ode_solves':ode,'ess_likelihood_evaluations':ee,'hard_rejections':hard,'mh_acceptances':acc,'exact_support_particles':int(np.sum(np.isfinite([ll(u,v,x,m,S,f,0.) for u,v in zip(mus,xi)])))};diagnostics.append(diag)
  checkpoint(root,stage,z,xi,mus,bases,logl,logw,{'beta':beta,'delta':delta,'phase':phase,'index':index,'status':'running'},diagnostics,rng);stage+=1
 w,ess,logw=normalize(logw);out={'status':'completed','particles':a.particles,'beta':beta,'delta':delta,'final_parameter_ess':ess,'final_parameter_ess_fraction':ess/a.particles,'zero_weight_fraction':float(np.mean(~np.isfinite(logw))),'posterior_mean':np.average(np.exp(z),axis=0,weights=w),'posterior_covariance':np.cov(np.exp(z),rowvar=False,aweights=w),'posterior_quantiles':np.quantile(np.exp(z[systematic(w,rng)]),[.05,.5,.95],axis=0),'diagnostics':diagnostics};dump(root/'results.json',out);dump(root/'progress.json',{'status':'completed','beta':beta,'delta':delta,'phase':phase,'index':index,'stage':stage-1,'diagnostics':diagnostics})
if __name__=='__main__':main()
