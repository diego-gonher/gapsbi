import json
from pathlib import Path
import h5py
import numpy as np
import torch
from gapsbi.evaluation.nre_c_campaign import (
    validate_nre_c_campaign_run,
    validate_oup_partial_reference_metadata,
)

def _fixture(path: Path, ids=range(10), finite=True):
 path.mkdir(); torch.save({'method':'nre_c_zero_imputation','seed':101},path/'model_checkpoint.pt')
 rows=[{'reference_index':i,'c2st_accuracy':.5,'posterior_mean_shift':.1,'covariance_trace_ratio':1.} for i in ids]
 (path/'reference_evaluation.json').write_text(json.dumps({'method':'nre_c_zero_imputation','seed':101,'posterior_target':'theta_given_x_obs_mask','reference_ids':list(ids),'selected_grid_resolution':800,'grid_convergence':[{}],'reference_metrics':rows}))
 with h5py.File(path/'reference_posterior_samples.h5','w') as h:
  h['reference_indices']=list(ids);h['theta_posterior']=np.ones((len(list(ids)),2,2))* (1 if finite else np.nan)

def test_complete_and_smoke_runs_are_distinguished(tmp_path):
 complete=tmp_path/'complete';_fixture(complete);assert validate_nre_c_campaign_run(complete)['status']=='complete'
 smoke=tmp_path/'smoke';_fixture(smoke,[0,1]);assert validate_nre_c_campaign_run(smoke)['status']=='incomplete'

def test_corrupt_and_mismatched_runs_fail(tmp_path):
 corrupt=tmp_path/'corrupt';_fixture(corrupt,finite=False);assert validate_nre_c_campaign_run(corrupt)['status']=='failed'
 mismatch=tmp_path/'mismatch';_fixture(mismatch);assert validate_nre_c_campaign_run(mismatch,expected_seed=202)['status']=='failed'


def test_oup_reference_metadata_requires_matching_mechanism_and_fraction():
 metadata = {'problem': 'oup', 'missingness_mechanism': 'mnar', 'mask': {'missing_fraction': .1}}
 validate_oup_partial_reference_metadata(metadata, mechanism='mnar', missing_fraction=.1)
 import pytest
 with pytest.raises(ValueError, match='missingness_mechanism'):
  validate_oup_partial_reference_metadata(metadata, mechanism='mcar', missing_fraction=.1)
