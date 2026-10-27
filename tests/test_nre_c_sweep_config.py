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


def test_nre_c_oup_mnar_sweep_matches_canonical_npe_dataset_and_rise_policy():
    config = yaml.safe_load(Path('experiments/nre_c_oup_mnar_sweep.yaml').read_text())
    assert config['mechanism'] == 'mnar'
    assert config['dataset_path'] == (
        'data/canonical_v1/oup/mnar/'
        'oup_mnar_self_censoring_mean_normalized_identity_eps010_seed123.h5'
    )
    assert config['partial_reference_path'] == 'references/partial_data_references_v1/oup_mnar_eps010_references.h5'
    assert config['seeds'] == [101, 202, 303, 404, 505]
    assert config['reference_indices'] == list(range(10))
    rise = config['method_overrides']['nre_c_probabilistic_learned_imputation']['rise']
    assert rise == {'lambda_np': 1.0, 'lambda_mask': 1.0, 'use_mask_head': True}
