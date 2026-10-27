"""Evaluate OUP NRE-C learned constants against partial-data references."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import h5py
import numpy as np
import torch
import yaml
from gapsbi.evaluation.reference_metrics import aggregate_final_reference_metrics, compute_reference_metrics
from gapsbi.methods.sbi_nre import evaluate_nre_a_oup_grid, load_nre_c_learned_constant_imputation_checkpoint, sample_nre_a_oup_grid
from gapsbi.preprocessing.scalers import transform_x_obs_with_fitted_scaler
from gapsbi.references import load_partial_data_reference_posteriors_hdf5

def _positions(observations, requested):
    ids = np.asarray(observations.get("reference_indices", np.arange(len(observations["x_obs"]))), dtype=int)
    wanted = ids if requested is None else np.asarray(requested, dtype=int); out=[]
    for reference_id in wanted:
        found=np.flatnonzero(ids == reference_id)
        if len(found) != 1: raise ValueError(f"Requested reference ID {reference_id} is not uniquely available.")
        out.append(int(found[0]))
    out=np.asarray(out,dtype=int); return out, ids[out]

def evaluate_checkpoint(path, config, *, reference_ids=None, repository_root=Path(".")):
    started=time.perf_counter(); checkpoint, estimator, imputer=load_nre_c_learned_constant_imputation_checkpoint(path, repository_root=repository_root)
    evaluation=config["evaluation"]; obs, reference_samples, metadata, _=load_partial_data_reference_posteriors_hdf5(evaluation["partial_reference_path"])
    if metadata.get("problem") != "oup" or metadata.get("missingness_mechanism") != "mcar" or float(metadata.get("mask",{}).get("missing_fraction",np.nan)) != .1: raise ValueError("Expected OUP MCAR-10% partial-data references.")
    positions, ids=_positions(obs, evaluation.get("reference_indices") if reference_ids is None else reference_ids)
    scaled=transform_x_obs_with_fitted_scaler(obs["x_obs"][positions],obs["mask"][positions],checkpoint["x_scaler"],checkpoint["x_scaling_metadata"]["transform"])
    with torch.no_grad(): conditions=imputer(scaled,torch.as_tensor(obs["mask"][positions],dtype=torch.float32)).numpy()
    resolutions=[int(x) for x in evaluation["grid_resolutions"]]; selected_resolution=int(evaluation["selected_grid_resolution"])
    if selected_resolution not in resolutions: raise ValueError("selected_grid_resolution must occur in grid_resolutions.")
    grids=[]; diagnostics=[]
    for reference_id,condition in zip(ids,conditions,strict=True):
        candidates=[evaluate_nre_a_oup_grid(estimator,theta_scaler=checkpoint["theta_scaler"],x_scaled=condition,resolution=r,batch_size=int(evaluation["grid_batch_size"])) for r in resolutions]
        selected=next(g for g in candidates if g["resolution"] == selected_resolution); grids.append(selected)
        diagnostics.append({"reference_index":int(reference_id),"resolutions":resolutions,"means":[g["mean"].tolist() for g in candidates],"covariance_traces":[float(np.trace(g["cov"])) for g in candidates],"boundary_masses":[g["boundary_mass"] for g in candidates],"mean_l2_deltas":[float(np.linalg.norm(g["mean"]-selected["mean"])) for g in candidates],"covariance_trace_deltas":[float(abs(np.trace(g["cov"])-np.trace(selected["cov"]))) for g in candidates]})
    samples=np.stack([sample_nre_a_oup_grid(g,num_samples=reference_samples.shape[1],seed=int(checkpoint["seed"])+40000+int(reference_id)) for reference_id,g in zip(ids,grids,strict=True)])
    metrics=evaluation["reference_metrics"]; rows=compute_reference_metrics(samples,reference_samples[positions],seed=int(checkpoint["seed"]),max_samples_per_observation=10000,c2st_max_samples_per_observation=int(metrics["c2st_max_samples_per_observation"]),c2st_n_folds=int(metrics["c2st_n_folds"]),c2st_hidden_layer_scale=int(metrics["c2st_hidden_layer_scale"]),c2st_max_iter=int(metrics["c2st_max_iter"]))
    for row,reference_id in zip(rows,ids,strict=True): row["reference_index"]=int(reference_id)
    result={"method":"nre_c_learned_constant_imputation","seed":int(checkpoint["seed"]),"posterior_target":"theta_given_x_obs_mask","imputation":"learned_constant_scaled_space","reference_artifact":str(evaluation["partial_reference_path"]),"reference_ids":ids.tolist(),"reference_metadata":metadata,"learned_constants_scaled":checkpoint["learned_constants_scaled"].tolist(),"grid_log_ratio_convention":"log_p_x_given_theta_minus_log_p_x","grid_prior":"uniform_raw_oup_box","selected_grid_resolution":selected_resolution,"inference_runtime_sec":time.perf_counter()-started,"grid_convergence":diagnostics,"reference_metrics":rows,"reference_metric_aggregate":aggregate_final_reference_metrics(rows)}
    output=Path(path).parent
    with h5py.File(output/"reference_posterior_samples.h5","w") as h:
        for key in ("theta_true","x_full","x_obs","mask"): h.create_dataset(key,data=obs[key][positions])
        h.create_dataset("reference_indices",data=ids); h.create_dataset("theta_posterior",data=samples); h.create_dataset("learned_constants_scaled",data=checkpoint["learned_constants_scaled"]); h.attrs["method"]="nre_c_learned_constant_imputation"; h.attrs["grid_resolution"]=selected_resolution; h.attrs["posterior_target"]="theta_given_x_obs_mask"
    (output/"reference_evaluation.json").write_text(json.dumps(result,indent=2)+"\n"); return result

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--config",type=Path,required=True); p.add_argument("--reference-indices",type=int,nargs="*",default=None); a=p.parse_args(); r=evaluate_checkpoint(a.checkpoint,yaml.safe_load(a.config.read_text()),reference_ids=a.reference_indices); print(json.dumps(r["reference_metric_aggregate"],sort_keys=True))
if __name__=="__main__": main()
