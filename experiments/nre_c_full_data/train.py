"""Train and evaluate the experimental low-budget full-data OUP NRE-C baseline."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import yaml
from gapsbi.checkpointing import DEFAULT_CHECKPOINT_NAME, save_model_checkpoint
from gapsbi.datasets import apply_train_val_sample_limits
from gapsbi.io import load_gapsbi_hdf5
from gapsbi.methods.sbi_nre import FixedSplitNRE_C, load_nre_c_full_data_checkpoint
from gapsbi.preprocessing.scalers import infer_theta_transform, infer_x_transform, make_scaled_theta_prior, scale_theta_train_val_test, scale_x_train_val_test, theta_scaling_metadata
from gapsbi.utils.seeding import set_all_seeds

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',type=Path,required=True); p.add_argument('--skip-reference-evaluation',action='store_true'); a=p.parse_args()
    config=yaml.safe_load(a.config.read_text()); d,meta=load_gapsbi_hdf5(config['dataset_path']); d=apply_train_val_sample_limits(d,config['max_train_samples'],config['max_val_samples'])
    tt,tv,_test,ts=scale_theta_train_val_test(d['train']['theta'],d['val']['theta'],d['test']['theta'],infer_theta_transform('oup'))
    xt,xv,_testx,xs,xmeta=scale_x_train_val_test(d['train']['x_full'],d['val']['x_full'],d['test']['x_full'],infer_x_transform('oup'))
    cfg=config['nre']; prior=make_scaled_theta_prior('oup',2,tt)
    for seed in config['seeds']:
        seed=int(seed); set_all_seeds(seed); out=Path(config['output_dir'])/f'seed_{seed}'; out.mkdir(parents=True,exist_ok=True)
        inf=FixedSplitNRE_C(prior=prior,classifier=cfg['classifier'],device=cfg['device'],show_progress_bars=False); inf.append_simulations(torch.cat((tt,tv)),torch.cat((xt,xv))); inf.set_fixed_train_val_split(len(tt),len(tv))
        start=time.perf_counter(); est=inf.train(num_classes=int(cfg['num_classes']),gamma=float(cfg['gamma']),training_batch_size=int(cfg['training_batch_size']),validation_fraction=.1,stop_after_epochs=int(cfg['stop_after_epochs']),max_num_epochs=int(cfg['max_num_epochs'])); runtime=time.perf_counter()-start
        h=inf.summary; train,val=[float(x) for x in h['training_loss']],[float(x) for x in h['validation_loss']]
        if not train or not np.isfinite(train).all() or not np.isfinite(val).all(): raise FloatingPointError('NRE-C losses are nonfinite.')
        with torch.no_grad(): finite=torch.isfinite(est.unnormalized_log_ratio(tv[:8],xv[:8])).all().item()
        if not finite: raise FloatingPointError('NRE-C log ratios are nonfinite.')
        ckpt=save_model_checkpoint(out/DEFAULT_CHECKPOINT_NAME,method='nre_c_full_data',problem='oup',seed=seed,config=config,dataset_path=Path(config['dataset_path']),theta_scaler=ts,theta_scaling_metadata=theta_scaling_metadata(infer_theta_transform('oup')),x_scaler=xs,x_scaling_metadata=xmeta,theta_dim=2,x_dim=25,density_estimator=est,extra={'config_path':str(a.config),'classifier':cfg['classifier'],'nre_c_num_classes':int(cfg['num_classes']),'nre_c_gamma':float(cfg['gamma']),'training_runtime_sec':runtime,'epochs_trained':len(train),'final_training_loss':train[-1],'final_validation_loss':val[-1],'best_validation_loss':min(val),'dataset_metadata':meta})
        _, reload_est=load_nre_c_full_data_checkpoint(ckpt)
        with torch.no_grad(): reloaded_finite=torch.isfinite(reload_est.unnormalized_log_ratio(tv[:8],xv[:8])).all().item()
        summary={'method':'nre_c_full_data','seed':seed,'num_train_examples':len(tt),'num_val_examples':len(tv),'training_runtime_sec':runtime,'epochs_trained':len(train),'final_training_loss':train[-1],'final_validation_loss':val[-1],'best_validation_loss':min(val),'nre_c_num_classes':int(cfg['num_classes']),'nre_c_gamma':float(cfg['gamma']),'validation_log_ratio_finite':bool(finite),'reloaded_log_ratio_finite':bool(reloaded_finite),'checkpoint_path':str(ckpt)}
        (out/'training_summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,sort_keys=True))
        if not a.skip_reference_evaluation:
            from evaluate import evaluate_checkpoint
            evaluate_checkpoint(ckpt,config)
if __name__=='__main__': main()
