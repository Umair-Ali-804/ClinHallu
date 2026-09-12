"""End-to-end: interrupt a real training run, restart it, and check the result.

This exercises the whole path a user hits when Colab disconnects mid-epoch --
real tokenizer, real DeBERTa (tiny), real optimizer, real checkpoint files, real
subprocess boundaries -- rather than a simulation of it. The reused fixture
builds a self-contained project on disk with no network access.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from clinhallu.config import load_config
from clinhallu.core.interrupt import InterruptGuard
from clinhallu.engine import trainer
from clinhallu.engine.checkpoints import (
    PHASE_IN_EPOCH,
    find_latest_checkpoint,
    load_checkpoint,
)
from clinhallu.experiments.runner import run_experiment

from test_offline_complete_pipeline import local_project  # noqa: F401  (pytest fixture)


def _multi_epoch_config(project: Path, epochs: int = 2) -> Path:
    path = project / "configs/base.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg["training"]["max_epochs"] = epochs
    cfg["training"]["early_stopping_patience"] = epochs + 1
    cfg["training"]["batch_size"] = 1
    cfg["training"]["gradient_accumulation_steps"] = 1
    cfg["checkpoints"]["save_every_steps"] = 1
    cfg["checkpoints"]["save_every_minutes"] = 0
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_training_stopped_mid_epoch_resumes_and_finishes(local_project, monkeypatch):  # noqa: F811
    config_path = _multi_epoch_config(local_project, epochs=2)
    run_experiment(config_path, prepare_only=True)
    cfg = load_config(config_path)

    # --- attempt 1: stop after a few optimizer steps of epoch 1 ------------ #
    guard = InterruptGuard(install=False)
    real_run_epoch = trainer.run_epoch
    steps = {"count": 0}

    def stopping_run_epoch(*args, **kwargs):
        hook = kwargs.get("checkpoint_hook")
        if hook is not None:

            def wrapped(batches_done, samples_done, accumulator):
                steps["count"] += 1
                hook(batches_done, samples_done, accumulator)
                if steps["count"] >= 3:
                    guard.trigger("test")

            kwargs["checkpoint_hook"] = wrapped
        return real_run_epoch(*args, **kwargs)

    monkeypatch.setattr(trainer, "run_epoch", stopping_run_epoch)
    first = trainer.train(cfg, resume=True, guard=guard)
    assert first["status"] == "interrupted"

    latest = Path(cfg["checkpoints"]["dir"]) / cfg["checkpoints"]["latest_filename"]
    payload = load_checkpoint(str(latest), torch.device("cpu"))
    assert payload["phase"] == PHASE_IN_EPOCH
    assert payload["epoch"] == 1
    assert 0 < payload["batches_done"] < 8  # stopped part-way through the epoch
    saved_batches = payload["batches_done"]

    # No best checkpoint yet: no epoch has been scored.
    assert not (Path(cfg["checkpoints"]["dir"]) / cfg["checkpoints"]["best_filename"]).is_file()

    # --- attempt 2: the same command, uninterrupted ------------------------ #
    monkeypatch.setattr(trainer, "run_epoch", real_run_epoch)
    second = trainer.train(load_config(config_path), resume=True)
    assert second["status"] == "completed"

    # It continued rather than starting over.
    history = (Path(cfg["results"]["training_log_csv"])).read_text().splitlines()
    assert len(history) == 3  # header + 2 epochs
    assert Path(second["checkpoint"]).is_file()
    metrics = json.loads(
        (Path(cfg["results"]["metrics_dir"]) / "validation_metrics.json").read_text()
    )
    assert all(v == v for v in metrics.values() if isinstance(v, float))  # no NaN
    assert saved_batches > 0


def test_resume_does_not_repeat_a_scored_epoch(local_project):  # noqa: F811
    """After epoch 1 is scored, a restart begins at epoch 2, not epoch 1."""
    config_path = _multi_epoch_config(local_project, epochs=3)
    cfg = load_config(config_path)
    run_experiment(config_path, prepare_only=True)

    # Train a single epoch by capping max_epochs, then raise the cap.
    short = load_config(config_path)
    short["training"]["max_epochs"] = 1
    trainer.train(short, resume=True)

    state_model = trainer.build_model(load_config(config_path))
    state = find_latest_checkpoint(
        cfg["checkpoints"]["dir"],
        cfg["checkpoints"]["latest_filename"],
        torch.device("cpu"),
        state_model,
    )
    assert state.start_epoch == 2
    assert not state.mid_epoch


def test_run_experiment_skips_training_when_artifacts_exist(local_project, monkeypatch):  # noqa: F811
    """The stage ledger short-circuits a completed run instead of retraining."""
    config_path = local_project / "configs/base.yaml"
    first = run_experiment(config_path)
    assert first["status"] == "evaluated"

    def forbidden(*args, **kwargs):
        raise AssertionError("A completed run must not launch any stage subprocess")

    monkeypatch.setattr("clinhallu.experiments.runner._run", forbidden)
    second = run_experiment(config_path)
    assert second["run_dir"] == first["run_dir"]

    ledger = json.loads((Path(first["run_dir"]) / "stage_ledger.json").read_text())
    assert ledger["units"]["train"]["status"] == "completed"
    assert ledger["units"]["evaluate"]["status"] == "completed"


def test_no_resume_starts_training_over(local_project, monkeypatch):  # noqa: F811
    config_path = _multi_epoch_config(local_project, epochs=1)
    cfg = load_config(config_path)
    run_experiment(config_path, prepare_only=True)
    trainer.train(load_config(config_path), resume=True)

    history_before = Path(cfg["results"]["training_log_csv"]).read_text()
    trainer.train(load_config(config_path), resume=False)
    history_after = Path(cfg["results"]["training_log_csv"]).read_text()
    # Fresh run: one epoch again rather than an appended second row.
    assert len(history_after.splitlines()) == len(history_before.splitlines()) == 2


@pytest.mark.parametrize("phase_steps", [1, 2])
def test_checkpoint_cadence_is_honoured(local_project, monkeypatch, phase_steps):  # noqa: F811
    """A save happens every configured number of optimizer steps."""
    config_path = _multi_epoch_config(local_project, epochs=1)
    cfg = load_config(config_path)
    cfg["checkpoints"]["save_every_steps"] = phase_steps
    cfg["checkpoints"]["save_every_minutes"] = 0
    run_experiment(config_path, prepare_only=True)

    saves = {"count": 0}
    real_save = trainer.save_checkpoint

    def counting_save(**kwargs):
        if kwargs.get("phase") == PHASE_IN_EPOCH:
            saves["count"] += 1
        return real_save(**kwargs)

    monkeypatch.setattr(trainer, "save_checkpoint", counting_save)
    trainer.train(cfg, resume=False)
    assert saves["count"] >= 1
