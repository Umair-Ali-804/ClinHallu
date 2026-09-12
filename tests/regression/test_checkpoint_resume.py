"""Checkpoints must survive a crash and describe exactly where to continue."""

import random

import numpy as np
import pytest
import torch

from clinhallu.engine.checkpoints import (
    PHASE_EPOCH_COMPLETE,
    PHASE_EPOCH_TRAINED,
    PHASE_IN_EPOCH,
    capture_rng_state,
    find_latest_checkpoint,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)


def _save(tmp_path, model, optimizer, **overrides):
    payload = dict(
        checkpoint_dir=str(tmp_path),
        filename="latest.pt",
        epoch=1,
        global_step=10,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state={},
        scaler_state=None,
        best_metric=0.5,
        patience_count=0,
    )
    payload.update(overrides)
    return save_checkpoint(**payload)


def test_round_trip_preserves_state_keys(tmp_path):
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    _save(tmp_path, model, optimizer)
    loaded = load_checkpoint(str(tmp_path / "latest.pt"), torch.device("cpu"))
    assert set(loaded["model_state"]) == set(model.state_dict())


def test_no_temporary_file_is_left_behind(tmp_path):
    model = torch.nn.Linear(3, 1)
    _save(tmp_path, model, torch.optim.AdamW(model.parameters()))
    assert not list(tmp_path.glob("*.tmp"))


def test_mid_epoch_state_round_trips(tmp_path):
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    _save(
        tmp_path,
        model,
        optimizer,
        phase=PHASE_IN_EPOCH,
        batches_done=96,
        samples_done=768,
        epoch_progress={"labels": [0, 1], "probabilities": [0.2, 0.9]},
    )
    state = find_latest_checkpoint(str(tmp_path), "latest.pt", torch.device("cpu"), model)
    assert state.resumed and state.mid_epoch
    # A half-finished epoch is continued, not counted as done.
    assert state.start_epoch == 1
    assert (state.batches_done, state.samples_done) == (96, 768)
    assert state.epoch_progress["probabilities"] == [0.2, 0.9]


def test_completed_epoch_advances_to_the_next_one(tmp_path):
    model = torch.nn.Linear(3, 1)
    _save(
        tmp_path,
        model,
        torch.optim.AdamW(model.parameters()),
        epoch=2,
        phase=PHASE_EPOCH_COMPLETE,
    )
    state = find_latest_checkpoint(str(tmp_path), "latest.pt", torch.device("cpu"), model)
    assert state.start_epoch == 3
    assert not state.mid_epoch


def test_trained_but_unscored_epoch_resumes_at_validation(tmp_path):
    model = torch.nn.Linear(3, 1)
    _save(
        tmp_path,
        model,
        torch.optim.AdamW(model.parameters()),
        epoch=2,
        phase=PHASE_EPOCH_TRAINED,
    )
    state = find_latest_checkpoint(str(tmp_path), "latest.pt", torch.device("cpu"), model)
    assert state.start_epoch == 2
    assert state.needs_validation_only


def test_corrupt_newest_checkpoint_falls_back_to_the_backup(tmp_path):
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    _save(tmp_path, model, optimizer, epoch=1, global_step=10)
    _save(tmp_path, model, optimizer, epoch=2, global_step=20)  # rotates epoch 1 to .bak
    assert (tmp_path / "latest.pt.bak").is_file()

    # Simulate a power cut during the write of the newest file.
    (tmp_path / "latest.pt").write_bytes(b"\x00\x01truncated")

    state = find_latest_checkpoint(str(tmp_path), "latest.pt", torch.device("cpu"), model)
    assert state.resumed
    assert state.global_step == 10  # one interval lost, not the whole run


def test_every_checkpoint_unreadable_raises_rather_than_silently_restarting(tmp_path):
    model = torch.nn.Linear(3, 1)
    (tmp_path / "latest.pt").write_bytes(b"garbage")
    (tmp_path / "latest.pt.bak").write_bytes(b"garbage")
    with pytest.raises(RuntimeError, match="unreadable"):
        find_latest_checkpoint(str(tmp_path), "latest.pt", torch.device("cpu"), model)


def test_missing_checkpoint_starts_from_scratch(tmp_path):
    model = torch.nn.Linear(3, 1)
    state = find_latest_checkpoint(str(tmp_path), "latest.pt", torch.device("cpu"), model)
    assert not state.resumed and state.start_epoch == 1


def test_rng_capture_and_restore_reproduce_the_same_stream():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    [random.random() for _ in range(5)]

    snapshot = capture_rng_state()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    # Advance every generator well past that point.
    [random.random() for _ in range(50)]
    np.random.rand(50)
    torch.rand(50)

    restore_rng_state(snapshot)
    assert (random.random(), float(np.random.rand()), float(torch.rand(1))) == expected


def test_checkpoint_loads_under_weights_only(tmp_path):
    """RNG state is stored as plain types so the safe loader path keeps working."""
    model = torch.nn.Linear(3, 1)
    _save(tmp_path, model, torch.optim.AdamW(model.parameters()))
    payload = torch.load(str(tmp_path / "latest.pt"), map_location="cpu", weights_only=True)
    assert payload["phase"] == PHASE_EPOCH_COMPLETE
