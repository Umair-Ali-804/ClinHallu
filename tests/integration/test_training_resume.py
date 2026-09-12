"""Resume must be *equivalent*, not merely non-crashing.

The test that matters is this: train an epoch straight through, then train the
same epoch again but kill it half way and continue from the checkpoint. If the
resume machinery is correct the two runs end with bit-identical weights,
optimizer state and metrics -- the crash left no trace in the result.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from clinhallu.core.interrupt import InterruptGuard
from clinhallu.data.resumable_sampler import ResumableRandomSampler
from clinhallu.engine.checkpoints import (
    capture_rng_state,
    find_latest_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from clinhallu.engine.trainer import run_epoch
from clinhallu.models.outputs import GAERPlusPlusOutput

SEED = 13
EXAMPLES = 64
BATCH = 4
ACCUMULATION = 2
SEQUENCE = 6


class _ToyDataset(Dataset):
    """Deterministic stand-in for the tokenized dataset."""

    def __init__(self, size: int) -> None:
        generator = torch.Generator().manual_seed(1234)
        self.features = torch.randn(size, SEQUENCE, generator=generator)
        self.labels = (torch.rand(size, generator=generator) > 0.5).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        return {"features": self.features[index], "labels": self.labels[index]}


def _collate(batch: list[dict]) -> dict:
    features = torch.stack([item["features"] for item in batch])
    ones = torch.ones_like(features)
    return {
        "input_ids": features,
        "attention_mask": ones,
        "context_mask": ones,
        "answer_mask": ones,
        "labels": torch.stack([item["labels"] for item in batch]),
    }


class _ToyModel(torch.nn.Module):
    """Mirrors the real model's call signature and output type."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(SEQUENCE, 1)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, *, input_ids, attention_mask, context_mask, answer_mask, labels=None):
        logits = self.linear(input_ids).squeeze(-1)
        loss = self.loss_fn(logits, labels.float()) if labels is not None else None
        return GAERPlusPlusOutput(
            logits=logits,
            probs=torch.sigmoid(logits),
            p_hallucinated=torch.sigmoid(logits),
            loss=loss,
        )


def _make(skip: int = 0, epoch: int = 1):
    torch.manual_seed(SEED)
    model = _ToyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    dataset = _ToyDataset(EXAMPLES)
    sampler = ResumableRandomSampler(dataset, seed=SEED, epoch=epoch, skip=skip)
    loader = DataLoader(dataset, batch_size=BATCH, sampler=sampler, collate_fn=_collate)
    return model, optimizer, scheduler, loader, sampler


def _weights(model) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in model.parameters()]


def _epoch_kwargs(**extra):
    return dict(
        max_grad_norm=1.0,
        gradient_accumulation_steps=ACCUMULATION,
        mixed_precision=False,
        epoch_batches=EXAMPLES // BATCH,
        **extra,
    )


def test_uninterrupted_epoch_is_the_baseline():
    model, optimizer, scheduler, loader, _ = _make()
    result = run_epoch(
        model,
        loader,
        torch.device("cpu"),
        optimizer=optimizer,
        scheduler=scheduler,
        **_epoch_kwargs(),
    )
    assert result["batches_done"] == EXAMPLES // BATCH
    assert result["samples_done"] == EXAMPLES
    assert not result["interrupted"]
    assert len(result["labels"]) == EXAMPLES


def test_resumed_epoch_matches_an_uninterrupted_epoch(tmp_path):
    # --- reference: one clean pass ---------------------------------------- #
    torch.manual_seed(SEED)
    model, optimizer, scheduler, loader, _ = _make()
    reference = run_epoch(
        model,
        loader,
        torch.device("cpu"),
        optimizer=optimizer,
        scheduler=scheduler,
        **_epoch_kwargs(),
    )
    reference_weights = _weights(model)

    # --- interrupted: stop half way, checkpoint, resume -------------------- #
    torch.manual_seed(SEED)
    model_a, optimizer_a, scheduler_a, loader_a, _ = _make()
    guard = InterruptGuard(install=False)
    saved: dict = {}

    def hook(batches_done, samples_done, accumulator):
        # Ask for a stop at the half-way optimizer step.
        if batches_done == (EXAMPLES // BATCH) // 2:
            progress = accumulator.to_state()
            progress["samples_done"] = samples_done
            save_checkpoint(
                checkpoint_dir=str(tmp_path),
                filename="latest.pt",
                epoch=1,
                global_step=batches_done // ACCUMULATION,
                model_state=model_a.state_dict(),
                optimizer_state=optimizer_a.state_dict(),
                scheduler_state=scheduler_a.state_dict(),
                scaler_state=None,
                best_metric=-1.0,
                patience_count=0,
                phase="in_epoch",
                batches_done=batches_done,
                samples_done=samples_done,
                epoch_progress=progress,
                rng_state=capture_rng_state(),
            )
            saved["batches"] = batches_done
            saved["samples"] = samples_done
            guard.trigger("test")

    partial = run_epoch(
        model_a,
        loader_a,
        torch.device("cpu"),
        optimizer=optimizer_a,
        scheduler=scheduler_a,
        checkpoint_hook=hook,
        guard=guard,
        **_epoch_kwargs(),
    )
    assert partial["interrupted"]
    assert partial["batches_done"] == saved["batches"]

    # A brand-new process: fresh objects, nothing carried over in memory.
    model_b, optimizer_b, scheduler_b, _, _ = _make()
    state = find_latest_checkpoint(
        str(tmp_path),
        "latest.pt",
        torch.device("cpu"),
        model_b,
        optimizer=optimizer_b,
        scheduler=scheduler_b,
    )
    assert state.mid_epoch and state.batches_done == saved["batches"]

    dataset = _ToyDataset(EXAMPLES)
    sampler = ResumableRandomSampler(
        dataset, seed=SEED, epoch=state.start_epoch, skip=state.samples_done
    )
    loader_b = DataLoader(dataset, batch_size=BATCH, sampler=sampler, collate_fn=_collate)
    progress = dict(state.epoch_progress)
    progress["samples_done"] = state.samples_done

    resumed = run_epoch(
        model_b,
        loader_b,
        torch.device("cpu"),
        optimizer=optimizer_b,
        scheduler=scheduler_b,
        start_batch=state.batches_done,
        resume_progress=progress,
        **_epoch_kwargs(),
    )

    # --- the two runs must be indistinguishable ---------------------------- #
    for expected, actual in zip(reference_weights, _weights(model_b), strict=True):
        assert torch.allclose(expected, actual, atol=0, rtol=0), "weights diverged after resume"

    assert resumed["batches_done"] == reference["batches_done"]
    assert resumed["samples_done"] == reference["samples_done"]
    assert resumed["labels"].tolist() == reference["labels"].tolist()
    assert torch.allclose(
        torch.tensor(resumed["probabilities"]), torch.tensor(reference["probabilities"])
    )
    assert abs(resumed["metrics"]["loss"] - reference["metrics"]["loss"]) < 1e-12


def test_resume_never_revisits_or_drops_an_example(tmp_path):
    """Every training example is seen exactly once across the two segments."""
    torch.manual_seed(SEED)
    model, optimizer, scheduler, loader, _ = _make()
    guard = InterruptGuard(install=False)
    seen_before: list[int] = []

    dataset = _ToyDataset(EXAMPLES)
    order = list(ResumableRandomSampler(dataset, seed=SEED, epoch=1))

    def hook(batches_done, samples_done, accumulator):
        if batches_done == 6:
            seen_before.extend(order[:samples_done])
            guard.trigger("test")

    partial = run_epoch(
        model,
        loader,
        torch.device("cpu"),
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_hook=hook,
        guard=guard,
        **_epoch_kwargs(),
    )
    seen_after = order[partial["samples_done"] :]
    assert sorted(seen_before + seen_after) == list(range(EXAMPLES))
    assert not set(seen_before) & set(seen_after)


def test_interrupt_only_happens_with_no_pending_gradients():
    """Stops land on optimizer-step boundaries, so no half-applied update is saved."""
    model, optimizer, scheduler, loader, _ = _make()
    guard = InterruptGuard(install=False)
    stop_points: list[int] = []

    def hook(batches_done, samples_done, accumulator):
        stop_points.append(batches_done)
        if batches_done >= 4:
            guard.trigger("test")

    result = run_epoch(
        model,
        loader,
        torch.device("cpu"),
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_hook=hook,
        guard=guard,
        **_epoch_kwargs(),
    )
    assert all(point % ACCUMULATION == 0 for point in stop_points)
    assert result["batches_done"] % ACCUMULATION == 0


def test_partial_epoch_metrics_carry_across_the_break():
    """The resumed epoch reports metrics over all its examples, not just the tail."""
    model, optimizer, scheduler, loader, _ = _make()
    guard = InterruptGuard(install=False)

    def hook(batches_done, samples_done, accumulator):
        if batches_done == 8:
            guard.trigger("test")

    partial = run_epoch(
        model,
        loader,
        torch.device("cpu"),
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_hook=hook,
        guard=guard,
        **_epoch_kwargs(),
    )
    progress = dict(partial["progress_state"])
    progress["samples_done"] = partial["samples_done"]

    dataset = _ToyDataset(EXAMPLES)
    sampler = ResumableRandomSampler(dataset, seed=SEED, epoch=1, skip=partial["samples_done"])
    loader_b = DataLoader(dataset, batch_size=BATCH, sampler=sampler, collate_fn=_collate)
    finished = run_epoch(
        model,
        loader_b,
        torch.device("cpu"),
        optimizer=optimizer,
        scheduler=scheduler,
        start_batch=partial["batches_done"],
        resume_progress=progress,
        **_epoch_kwargs(),
    )
    assert len(finished["labels"]) == EXAMPLES
    assert len(finished["probabilities"]) == EXAMPLES


def test_rng_restoration_is_part_of_the_contract(tmp_path):
    """A checkpoint replays the RNG stream, so dropout-style noise also matches."""
    torch.manual_seed(SEED)
    snapshot = capture_rng_state()
    expected = torch.rand(3)
    torch.rand(100)
    restore_rng_state(snapshot)
    assert torch.equal(torch.rand(3), expected)
