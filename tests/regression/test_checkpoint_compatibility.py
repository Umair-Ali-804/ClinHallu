import torch

from clinhallu.engine.checkpoints import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip_preserves_state_keys(tmp_path):
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    save_checkpoint(
        str(tmp_path),
        "test.pt",
        epoch=1,
        global_step=2,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state={},
        scaler_state=None,
        best_metric=0.8,
        patience_count=0,
    )
    loaded = load_checkpoint(str(tmp_path / "test.pt"), torch.device("cpu"))
    assert set(loaded["model_state"]) == set(model.state_dict())
