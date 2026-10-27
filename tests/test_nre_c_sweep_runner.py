import importlib.util
from pathlib import Path

import pytest
import torch

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/run_nre_c_oup_mcar_sweep.py"
_SPEC = importlib.util.spec_from_file_location("nre_c_oup_mcar_sweep", _MODULE_PATH)
sweep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sweep)


def test_campaign_matrix_and_materialized_budget_contract():
    matrix = sweep.load_matrix()
    configurations = sweep.jobs(matrix)
    assert len(configurations) == 75
    assert len({sweep.job_id(job) for job in configurations}) == 75
    expected = {
        "low": (900, 100, 100),
        "mid": (9000, 1000, 256),
        "high": (None, None, 256),
    }
    for budget, values in expected.items():
        job = next(job for job in configurations if job["budget"] == budget)
        config = sweep.materialize_config(job, matrix)
        assert (config["max_train_samples"], config["max_val_samples"], config["nre"]["training_batch_size"]) == values
        assert config["evaluation"]["reference_indices"] == list(range(10))
        assert config["evaluation"]["partial_reference_path"] == matrix["partial_reference_path"]
        assert config["seeds"] == [job["seed"]]


def test_mnar_matrix_materializes_isolated_paths_and_rise_override():
    matrix = sweep.load_matrix(Path("experiments/nre_c_oup_mnar_sweep.yaml"))
    configurations = sweep.jobs(matrix)
    assert len(configurations) == 75
    job = next(job for job in configurations if job["method"] == "nre_c_probabilistic_learned_imputation" and job["budget"] == "low")
    config = sweep.materialize_config(job, matrix)
    assert config["mechanism"] == "mnar"
    assert config["missing_fraction"] == 0.1
    assert config["output_dir"].endswith("/oup/mnar/oup_mnar_eps010")
    assert config["evaluation"]["partial_reference_path"] == matrix["partial_reference_path"]
    assert config["evaluation"]["reference_indices"] == list(range(10))
    assert config["rise"]["lambda_np"] == 1.0
    assert config["rise"]["use_mask_head"] is True


def test_matching_smoke_checkpoint_is_reused_and_evaluated(monkeypatch, tmp_path):
    matrix = sweep.load_matrix()
    job = next(job for job in sweep.jobs(matrix) if job["method"] == "nre_c_zero_imputation" and job["budget"] == "low" and job["seed"] == 101)
    config = sweep.materialize_config(job, matrix)
    folder = tmp_path / "seed_101"
    folder.mkdir()
    checkpoint_path = folder / "model_checkpoint.pt"
    torch.save({
        "method": job["method"], "problem": "oup", "seed": 101,
        "dataset_path": config["dataset_path"], "config": config,
        "density_estimator_state_dict": {"weight": torch.ones(1)},
    }, checkpoint_path)
    (folder / "reference_evaluation.json").write_text('{"reference_ids": [0, 1]}')
    (folder / "reference_posterior_samples.h5").touch()
    monkeypatch.setattr(sweep, "run_dir", lambda _job, _matrix: folder)
    monkeypatch.setattr(sweep, "validate_nre_c_campaign_run", lambda *args, **kwargs: {"status": "incomplete", "reason": "expected IDs 0-9, found [0, 1]"})
    status = sweep.current_status(job, matrix)
    assert status["status"] == "incomplete"
    assert "reused" in status["reason"]
    train, evaluate = sweep.commands_for(job, matrix, tmp_path / "run.yaml")
    assert train is None
    assert evaluate[evaluate.index("--checkpoint") + 1] == str(checkpoint_path)


def test_mismatched_checkpoint_is_failed_not_overwritten(monkeypatch, tmp_path):
    matrix = sweep.load_matrix()
    job = next(job for job in sweep.jobs(matrix) if job["method"] == "nre_c_zero_imputation" and job["budget"] == "low" and job["seed"] == 101)
    config = sweep.materialize_config(job, matrix)
    folder = tmp_path / "seed_101"
    folder.mkdir()
    torch.save({"method": job["method"], "problem": "oup", "seed": 101,
                "dataset_path": config["dataset_path"], "config": {**config, "max_train_samples": 12},
                "density_estimator_state_dict": {"weight": torch.ones(1)}}, folder / "model_checkpoint.pt")
    monkeypatch.setattr(sweep, "run_dir", lambda _job, _matrix: folder)
    monkeypatch.setattr(sweep, "validate_nre_c_campaign_run", lambda *args, **kwargs: {"status": "incomplete", "reason": "incomplete"})
    assert sweep.current_status(job, matrix)["status"] == "failed"


def test_checkpoint_without_full_evaluation_resumes_at_evaluation(monkeypatch, tmp_path):
    matrix = sweep.load_matrix()
    job = next(job for job in sweep.jobs(matrix) if job["method"] == "nre_c_zero_imputation" and job["budget"] == "low" and job["seed"] == 101)
    config = sweep.materialize_config(job, matrix)
    folder = tmp_path / "seed_101"
    folder.mkdir()
    torch.save({"method": job["method"], "problem": "oup", "seed": 101,
                "dataset_path": config["dataset_path"], "config": config,
                "density_estimator_state_dict": {"weight": torch.ones(1)}}, folder / "model_checkpoint.pt")
    monkeypatch.setattr(sweep, "run_dir", lambda _job, _matrix: folder)
    monkeypatch.setattr(sweep, "validate_nre_c_campaign_run", lambda *args, **kwargs: {"status": "incomplete", "reason": "missing evaluation"})
    assert sweep.current_status(job, matrix)["status"] == "incomplete"
    train, _evaluate = sweep.commands_for(job, matrix, tmp_path / "run.yaml")
    assert train is None


def test_orphaned_evaluation_is_failed_and_preserved(monkeypatch, tmp_path):
    matrix = sweep.load_matrix()
    job = next(job for job in sweep.jobs(matrix) if job["method"] == "nre_c_zero_imputation" and job["budget"] == "low" and job["seed"] == 101)
    folder = tmp_path / "seed_101"
    folder.mkdir()
    (folder / "reference_evaluation.json").write_text("{}")
    monkeypatch.setattr(sweep, "run_dir", lambda _job, _matrix: folder)
    monkeypatch.setattr(sweep, "validate_nre_c_campaign_run", lambda *args, **kwargs: {"status": "incomplete", "reason": "missing checkpoint"})
    assert sweep.current_status(job, matrix)["status"] == "failed"


def test_validator_confirmed_complete_run_is_skipped(monkeypatch, tmp_path):
    matrix = sweep.load_matrix()
    job = next(job for job in sweep.jobs(matrix) if job["method"] == "nre_c_zero_imputation" and job["budget"] == "low" and job["seed"] == 101)
    config = sweep.materialize_config(job, matrix)
    folder = tmp_path / "seed_101"
    folder.mkdir()
    torch.save({"method": job["method"], "problem": "oup", "seed": 101,
                "dataset_path": config["dataset_path"], "config": config,
                "density_estimator_state_dict": {"weight": torch.ones(1)}}, folder / "model_checkpoint.pt")
    monkeypatch.setattr(sweep, "run_dir", lambda _job, _matrix: folder)
    monkeypatch.setattr(sweep, "validate_nre_c_campaign_run", lambda *args, **kwargs: {"status": "complete", "reason": "all references validated"})
    assert sweep.current_status(job, matrix)["status"] == "complete"


def test_campaign_lock_prevents_duplicate_process(monkeypatch, tmp_path):
    monkeypatch.setattr(sweep, "CAMPAIGN", tmp_path / "campaign")
    lock = sweep._acquire_lock()
    with pytest.raises(RuntimeError, match="already active"):
        sweep._acquire_lock()
    lock.unlink()
