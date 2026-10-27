from pathlib import Path
import yaml

def test_nre_c_oup_mcar_sweep_matrix_matches_tracked_npe_budgets():
    config=yaml.safe_load(Path('experiments/nre_c_oup_mcar_sweep.yaml').read_text())
    assert config['seeds']==[101,202,303,404,505]
    assert config['reference_indices']==list(range(10))
    assert config['budgets']['low']=={'max_train_samples':900,'max_val_samples':100,'training_batch_size':100}
    assert config['budgets']['mid']=={'max_train_samples':9000,'max_val_samples':1000,'training_batch_size':256}
    assert config['budgets']['high']['max_train_samples'] is None
    assert len(config['methods'])==5
