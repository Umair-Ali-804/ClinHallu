"""Training engine for GAER++ and the backward-compatible legacy model.

Training is crash-safe at *optimizer-step* granularity. A checkpoint is written
every ``checkpoints.save_every_steps`` optimizer steps and at least every
``checkpoints.save_every_minutes`` minutes, always at a boundary where no
gradients are pending. Re-running the same command continues from that point:
the same epoch, the same position in that epoch's shuffled order, the same
optimizer/scheduler/AMP state, the same RNG streams, and the same partially
accumulated epoch metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from clinhallu.config import load_config
from clinhallu.core.interrupt import INTERRUPTED_EXIT_CODE, InterruptGuard
from clinhallu.core.logging import configure_logging
from clinhallu.core.seeding import set_seed
from clinhallu.data.dataset import build_dataloaders_v2
from clinhallu.models import HKGFusionModelV2, build_model

from .calibration import calibrate_validation, save_calibration
from .checkpoints import (
    PHASE_EPOCH_COMPLETE,
    PHASE_EPOCH_TRAINED,
    PHASE_IN_EPOCH,
    CheckpointState,
    capture_rng_state,
    export_model_safetensors,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from .metrics import compute_binary_metrics, compute_ece
from .thresholding import search_threshold
from .validation import autocast_for, scalar_logits

logger = logging.getLogger(__name__)

DEFAULT_SAVE_EVERY_STEPS = 200
DEFAULT_SAVE_EVERY_MINUTES = 10.0


def _build_llrd_param_groups(
    model: HKGFusionModelV2,
    base_lr: float,
    decay: float,
    weight_decay: float,
) -> List[Dict]:
    encoder_named = list(model.encoder.named_parameters())
    other_named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.")
    ]
    layer_ids = set()
    for name, _ in encoder_named:
        parts = name.split(".")
        for index, part in enumerate(parts):
            if part == "layer" and index + 1 < len(parts) and parts[index + 1].isdigit():
                layer_ids.add(int(parts[index + 1]))
    max_layer = max(layer_ids) if layer_ids else 0

    def learning_rate(name: str) -> float:
        parts = name.split(".")
        for index, part in enumerate(parts):
            if part == "layer" and index + 1 < len(parts) and parts[index + 1].isdigit():
                return base_lr * (decay ** (max_layer - int(parts[index + 1])))
        return base_lr * (decay ** (max_layer + 1))

    grouped: Dict[float, list] = {}
    for name, parameter in encoder_named:
        if parameter.requires_grad:
            grouped.setdefault(learning_rate(name), []).append(parameter)
    result = [
        {"params": parameters, "lr": lr, "weight_decay": weight_decay}
        for lr, parameters in grouped.items()
    ]
    remaining = [parameter for _, parameter in other_named if parameter.requires_grad]
    if remaining:
        result.append({"params": remaining, "lr": base_lr, "weight_decay": weight_decay})
    return result


def _build_two_group_param_groups(
    model: torch.nn.Module,
    encoder_lr: float,
    head_lr: float,
    weight_decay: float,
) -> List[Dict]:
    encoder = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("encoder.") and parameter.requires_grad
    ]
    head = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.") and parameter.requires_grad
    ]
    groups = []
    if encoder:
        groups.append({"params": encoder, "lr": encoder_lr, "weight_decay": weight_decay})
    if head:
        groups.append({"params": head, "lr": head_lr, "weight_decay": weight_decay})
    return groups


def _log_parameter_counts(model: torch.nn.Module) -> None:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    lora = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name
    )
    head = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("encoder.")
    )
    logger.info(
        "Parameters: total=%s trainable=%s (%.3f%%) LoRA=%s GAER/head=%s",
        f"{total:,}",
        f"{trainable:,}",
        100.0 * trainable / max(total, 1),
        f"{lora:,}",
        f"{head:,}",
    )


def _make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except AttributeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=(device.type == "cuda")) for key, value in batch.items()
    }


def _optimizer_step(model, optimizer, scheduler, scaler, max_grad_norm: float) -> None:
    if scaler is not None and scaler.is_enabled():
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)


class EpochAccumulator:
    """Running labels/scores/loss for one epoch, serialisable into a checkpoint.

    Mid-epoch resume has to restore these too, otherwise the epoch's reported
    metrics would only describe the batches seen after the crash.
    """

    def __init__(self, state: Optional[dict] = None) -> None:
        state = state or {}
        self.labels: list[int] = [int(value) for value in state.get("labels", [])]
        self.probabilities: list[float] = [float(v) for v in state.get("probabilities", [])]
        self.logits: list[float] = [float(value) for value in state.get("logits", [])]
        self.total_loss: float = float(state.get("total_loss", 0.0))
        self.batches_with_loss: int = int(state.get("batches_with_loss", 0))

    def to_state(self) -> dict:
        return {
            "labels": self.labels,
            "probabilities": self.probabilities,
            "logits": self.logits,
            "total_loss": self.total_loss,
            "batches_with_loss": self.batches_with_loss,
        }

    def finalize(self) -> dict:
        predictions = [int(value >= 0.5) for value in self.probabilities]
        metrics = (
            compute_binary_metrics(self.labels, predictions, self.probabilities)
            if self.labels
            else {}
        )
        metrics["loss"] = self.total_loss / max(self.batches_with_loss, 1)
        return {
            "metrics": metrics,
            "labels": np.asarray(self.labels, dtype=np.int64),
            "probabilities": np.asarray(self.probabilities, dtype=np.float64),
            "logits": np.asarray(self.logits, dtype=np.float32),
        }


def run_epoch(
    model,
    loader,
    device: torch.device,
    *,
    optimizer=None,
    scheduler=None,
    scaler=None,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    mixed_precision: bool = False,
    description: str = "epoch",
    epoch_batches: int | None = None,
    start_batch: int = 0,
    resume_progress: Optional[dict] = None,
    checkpoint_hook=None,
    guard: Optional[InterruptGuard] = None,
) -> dict:
    """Run one train or validation pass and retain its logits.

    Returning logits means checkpoint selection does not require a second
    validation pass.

    Resume-related arguments:
        epoch_batches: batches in a *complete* epoch. When the loader has been
            shortened for a mid-epoch resume, gradient-accumulation group sizes
            must still be computed against the full epoch.
        start_batch: batches already consumed before this call.
        resume_progress: serialised :class:`EpochAccumulator` state to continue.
        checkpoint_hook: called at optimizer-step boundaries as
            ``hook(batches_done, samples_done, accumulator)``.
        guard: polled at step boundaries; when triggered the pass returns early
            with ``interrupted=True``.
    """

    training = optimizer is not None
    model.train(training)
    accumulator = EpochAccumulator(resume_progress)
    pending_step = False
    total_batches = epoch_batches if epoch_batches is not None else start_batch + len(loader)
    batches_done = start_batch
    samples_done = int((resume_progress or {}).get("samples_done", 0))
    interrupted = False

    if training:
        optimizer.zero_grad(set_to_none=True)

    progress = tqdm(
        loader,
        desc=description,
        leave=False,
        initial=start_batch,
        total=total_batches,
    )
    for local_step, raw_batch in enumerate(progress):
        # Index within the full epoch, so accumulation groups stay aligned
        # across a resume.
        step = start_batch + local_step
        batch = _move_batch(raw_batch, device)
        labels = batch.pop("labels", None)
        batch_size = int(batch["input_ids"].size(0))
        with torch.set_grad_enabled(training):
            with autocast_for(device, mixed_precision):
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    context_mask=batch["context_mask"],
                    answer_mask=batch["answer_mask"],
                    labels=labels,
                )

        if training and output.loss is not None:
            group_start = (step // gradient_accumulation_steps) * gradient_accumulation_steps
            group_size = min(gradient_accumulation_steps, total_batches - group_start)
            scaled_loss = output.loss / group_size
            if scaler is not None and scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            pending_step = True

        if output.loss is not None:
            accumulator.total_loss += float(output.loss.detach().item())
            accumulator.batches_with_loss += 1
        if labels is not None:
            accumulator.labels.extend(labels.detach().cpu().tolist())
            accumulator.probabilities.extend(output.p_hallucinated.detach().float().cpu().tolist())
            accumulator.logits.extend(scalar_logits(output).detach().float().cpu().tolist())

        batches_done = step + 1
        samples_done += batch_size

        if training and pending_step and batches_done % gradient_accumulation_steps == 0:
            _optimizer_step(model, optimizer, scheduler, scaler, max_grad_norm)
            pending_step = False
            # No gradients are pending here, so the state on disk is a exact
            # description of the run: the safest possible place to checkpoint.
            if checkpoint_hook is not None:
                checkpoint_hook(batches_done, samples_done, accumulator)
            if guard is not None and guard.triggered:
                interrupted = True
                break

    progress.close()

    if training and pending_step and not interrupted:
        _optimizer_step(model, optimizer, scheduler, scaler, max_grad_norm)

    result = accumulator.finalize()
    result["interrupted"] = interrupted
    result["batches_done"] = batches_done
    result["samples_done"] = samples_done
    result["progress_state"] = accumulator.to_state()
    return result


def _ensure_processed_splits(processed_dir: str) -> None:
    for filename in ("train.jsonl", "validation.jsonl"):
        path = Path(processed_dir) / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"Missing processed split {path}. Run 'clinhallu prepare' first."
            )


def _write_history(path: str | Path, rows: list[dict]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(
        {key for row in rows for key in row},
        key=lambda key: (key != "epoch", key),
    )
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)


def _read_history(path: str | Path) -> list[dict]:
    source = Path(path)
    if not source.is_file():
        return []
    with source.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _merge_history(from_checkpoint: list[dict], from_csv: list[dict]) -> list[dict]:
    """Prefer the checkpoint's history; fall back to the CSV for older runs.

    The CSV is rewritten after every epoch and the checkpoint carries the same
    rows, so either source is usually complete. Keeping whichever has more
    epochs avoids losing a row if one write happened to be the crash victim.
    """

    if len(from_checkpoint) >= len(from_csv):
        return [dict(row) for row in from_checkpoint]
    return [dict(row) for row in from_csv]


def _resumable_train_sampler(loader):
    """Return the loader's sampler when it supports mid-epoch continuation."""
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch") and hasattr(sampler, "set_skip"):
        return sampler
    return None


def train(
    cfg: dict,
    *,
    resume: bool = True,
    smoke_test: bool = False,
    guard: Optional[InterruptGuard] = None,
) -> dict:
    set_seed(int(cfg["training"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["training"]["mixed_precision"] and device.type == "cuda")
    _ensure_processed_splits(cfg["data"]["processed_dir"])

    revision = cfg["model"].get("revision")
    tokenizer_kwargs = {"revision": revision} if revision else {}
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder"], **tokenizer_kwargs)
    train_loader, validation_loader = build_dataloaders_v2(
        cfg,
        tokenizer,
        device,
        smoke_test=smoke_test,
        smoke_test_n=int(cfg["training"].get("smoke_test_examples", 1000)),
    )
    model = build_model(cfg).to(device)
    _log_parameter_counts(model)

    model_type = cfg["model"]["type"]
    base_lr = float(cfg["training"]["learning_rate"])
    weight_decay = float(cfg["training"]["weight_decay"])
    if model_type == "gaer_plus_plus":
        parameter_groups = _build_two_group_param_groups(
            model,
            base_lr,
            float(cfg["training"].get("head_learning_rate", 1e-4)),
            weight_decay,
        )
    elif cfg["training"].get("llrd_decay"):
        parameter_groups = _build_llrd_param_groups(
            model,
            base_lr,
            float(cfg["training"]["llrd_decay"]),
            weight_decay,
        )
    else:
        parameter_groups = [
            {
                "params": [
                    parameter for parameter in model.parameters() if parameter.requires_grad
                ],
                "lr": base_lr,
                "weight_decay": weight_decay,
            }
        ]

    optimizer = AdamW(parameter_groups, lr=base_lr, weight_decay=weight_decay)
    accumulation = int(cfg["training"]["gradient_accumulation_steps"])
    max_epochs = 1 if smoke_test else int(cfg["training"]["max_epochs"])
    train_sampler = _resumable_train_sampler(train_loader)
    epoch_batches = len(train_loader)
    steps_per_epoch = max(math.ceil(epoch_batches / accumulation), 1)
    total_steps = steps_per_epoch * max_epochs
    warmup_steps = int(total_steps * float(cfg["training"]["warmup_ratio"]))
    scheduler_factory = (
        get_cosine_schedule_with_warmup
        if model_type == "gaer_plus_plus"
        else get_linear_schedule_with_warmup
    )
    scheduler = scheduler_factory(optimizer, warmup_steps, total_steps)
    scaler = _make_scaler(use_amp)

    checkpoint_dir = cfg["checkpoints"]["dir"]
    best_filename = cfg["checkpoints"]["best_filename"]
    latest_filename = cfg["checkpoints"]["latest_filename"]
    save_every_steps = int(cfg["checkpoints"].get("save_every_steps", DEFAULT_SAVE_EVERY_STEPS))
    save_every_seconds = (
        float(cfg["checkpoints"].get("save_every_minutes", DEFAULT_SAVE_EVERY_MINUTES)) * 60.0
    )

    state = CheckpointState()
    if resume:
        state = find_latest_checkpoint(
            checkpoint_dir,
            latest_filename,
            device,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
    best_metric = state.best_metric
    patience = state.patience_count
    start_epoch = state.start_epoch
    global_step = state.global_step

    history = (
        _merge_history(state.history, _read_history(cfg["results"]["training_log_csv"]))
        if resume
        else []
    )

    checkpoint_extra = {
        "seed": int(cfg["training"]["seed"]),
        "model_type": model_type,
        "ablation_mode": cfg["model"].get("ablation_mode"),
    }

    def persist(
        *,
        epoch: int,
        phase: str,
        batches_done: int,
        samples_done: int,
        epoch_progress: dict | None,
        step: int,
        is_best: bool = False,
    ) -> None:
        payload = dict(
            checkpoint_dir=checkpoint_dir,
            epoch=epoch,
            global_step=step,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            scaler_state=scaler.state_dict() if scaler is not None else None,
            best_metric=best_metric,
            patience_count=patience,
            extra=checkpoint_extra,
            phase=phase,
            batches_done=batches_done,
            samples_done=samples_done,
            epoch_progress=epoch_progress or {},
            history=history,
            rng_state=capture_rng_state(),
        )
        save_checkpoint(filename=latest_filename, **payload)
        if is_best:
            save_checkpoint(filename=best_filename, **payload)

    stopped_early = False
    interrupted = False

    for epoch in range(start_epoch, max_epochs + 1):
        resume_here = state.resumed and epoch == state.start_epoch
        skip_batches = state.batches_done if (resume_here and state.mid_epoch) else 0
        skip_samples = state.samples_done if (resume_here and state.mid_epoch) else 0
        resume_progress = dict(state.epoch_progress) if (resume_here and state.mid_epoch) else None
        if resume_progress is not None:
            resume_progress["samples_done"] = skip_samples

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
            train_sampler.set_skip(skip_samples)
        elif skip_batches:
            logger.warning(
                "This DataLoader cannot skip examples, so epoch %d restarts from its "
                "beginning. Progress within the epoch was not lost from the model, "
                "but those examples will be seen twice.",
                epoch,
            )
            skip_batches, skip_samples, resume_progress = 0, 0, None

        # Phase 1: the training pass, unless a checkpoint says it already finished.
        if resume_here and state.needs_validation_only:
            logger.info("Skipping the epoch %d training pass; it completed before the stop.", epoch)
            training = {
                "metrics": EpochAccumulator(state.epoch_progress).finalize()["metrics"],
                "interrupted": False,
            }
            global_step = state.global_step
        else:
            last_save = time.monotonic()
            steps_since_save = 0

            def make_checkpoint_hook(current_epoch: int):
                """Bind the epoch explicitly so the closure cannot drift."""

                def checkpoint_hook(batches_done: int, samples_done: int, accumulator) -> None:
                    nonlocal global_step, last_save, steps_since_save
                    global_step += 1
                    steps_since_save += 1
                    due = steps_since_save >= save_every_steps or (
                        save_every_seconds > 0
                        and time.monotonic() - last_save >= save_every_seconds
                    )
                    if guard is not None and guard.triggered:
                        due = True
                    if not due:
                        return
                    progress_state = accumulator.to_state()
                    progress_state["samples_done"] = samples_done
                    persist(
                        epoch=current_epoch,
                        phase=PHASE_IN_EPOCH,
                        batches_done=batches_done,
                        samples_done=samples_done,
                        epoch_progress=progress_state,
                        step=global_step,
                    )
                    last_save = time.monotonic()
                    steps_since_save = 0
                    logger.info(
                        "Checkpoint written mid-epoch %d at batch %d/%d (step %d).",
                        current_epoch,
                        batches_done,
                        epoch_batches,
                        global_step,
                    )

                return checkpoint_hook

            training = run_epoch(
                model,
                train_loader,
                device,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                max_grad_norm=float(cfg["training"]["max_grad_norm"]),
                gradient_accumulation_steps=accumulation,
                mixed_precision=use_amp,
                description=f"train e{epoch}",
                epoch_batches=epoch_batches,
                start_batch=skip_batches,
                resume_progress=resume_progress,
                checkpoint_hook=make_checkpoint_hook(epoch),
                guard=guard,
            )

            if training["interrupted"]:
                progress_state = dict(training["progress_state"])
                progress_state["samples_done"] = training["samples_done"]
                persist(
                    epoch=epoch,
                    phase=PHASE_IN_EPOCH,
                    batches_done=training["batches_done"],
                    samples_done=training["samples_done"],
                    epoch_progress=progress_state,
                    step=global_step,
                )
                logger.warning(
                    "Stopped during epoch %d at batch %d/%d. Re-run the same command "
                    "to continue from here.",
                    epoch,
                    training["batches_done"],
                    epoch_batches,
                )
                interrupted = True
                break

            # The training pass is complete but unscored. Recording that fact
            # means an interruption during validation does not repeat the epoch.
            persist(
                epoch=epoch,
                phase=PHASE_EPOCH_TRAINED,
                batches_done=epoch_batches,
                samples_done=training["samples_done"],
                epoch_progress=training["progress_state"],
                step=global_step,
            )

        # Phase 2: validation and checkpoint selection.
        validation = run_epoch(
            model,
            validation_loader,
            device,
            mixed_precision=use_amp,
            description=f"validation e{epoch}",
        )
        threshold, optimized_f1 = search_threshold(
            validation["probabilities"], validation["labels"]
        )
        validation["metrics"]["f1_at_0.5"] = validation["metrics"].get("f1", 0.0)
        validation["metrics"]["f1_threshold_optimized"] = optimized_f1
        validation["metrics"]["threshold_this_epoch"] = threshold

        row = {"epoch": epoch}
        row.update(
            {
                f"train_{key}": value
                for key, value in training["metrics"].items()
                if not isinstance(value, list)
            }
        )
        row.update(
            {
                f"validation_{key}": value
                for key, value in validation["metrics"].items()
                if not isinstance(value, list)
            }
        )
        history = [item for item in history if int(float(item.get("epoch", -1))) != epoch]
        history.append(row)
        history.sort(key=lambda item: int(float(item.get("epoch", 0))))
        _write_history(cfg["results"]["training_log_csv"], history)

        logger.info(
            "Epoch %d train_loss=%.4f validation_loss=%.4f "
            "validation_f1@0.5=%.4f validation_f1@tau*=%.4f tau*=%.3f",
            epoch,
            training["metrics"]["loss"],
            validation["metrics"]["loss"],
            validation["metrics"].get("f1_at_0.5", 0.0),
            optimized_f1,
            threshold,
        )

        monitor = optimized_f1
        is_best = monitor > best_metric
        if is_best:
            best_metric, patience = monitor, 0
        else:
            patience += 1

        persist(
            epoch=epoch,
            phase=PHASE_EPOCH_COMPLETE,
            batches_done=epoch_batches,
            samples_done=0,
            epoch_progress={},
            step=global_step,
            is_best=is_best,
        )
        # Consumed: later epochs start fresh rather than inheriting this state.
        state = CheckpointState()

        if patience >= int(cfg["training"]["early_stopping_patience"]):
            logger.info("Early stopping at epoch %d", epoch)
            stopped_early = True
            break

        if guard is not None and guard.triggered:
            logger.warning(
                "Stopped cleanly after epoch %d. Re-run the same command to continue.", epoch
            )
            interrupted = True
            break

    if interrupted:
        return {
            "status": "interrupted",
            "checkpoint": str(Path(checkpoint_dir) / latest_filename),
            "resume_hint": "Re-run the same command; training continues from this checkpoint.",
        }

    best_path = Path(checkpoint_dir) / best_filename
    if not best_path.is_file():
        raise RuntimeError(f"Training completed without a best checkpoint: {best_path}")
    payload = load_checkpoint(str(best_path), device)
    model.load_state_dict(payload["model_state"], strict=True)

    final_validation = run_epoch(
        model,
        validation_loader,
        device,
        mixed_precision=use_amp,
        description="calibration",
    )
    logits = torch.from_numpy(final_validation["logits"]).float()
    labels = torch.from_numpy(final_validation["labels"]).float()
    temperature, threshold, validation_f1 = calibrate_validation(logits, labels)
    calibration = save_calibration(cfg["calibration"]["dir"], temperature, threshold, validation_f1)
    calibrated_probabilities = torch.sigmoid(logits / temperature).numpy()
    calibrated_predictions = (calibrated_probabilities >= threshold).astype(int)
    validation_metrics = compute_binary_metrics(
        labels.int().tolist(),
        calibrated_predictions.tolist(),
        calibrated_probabilities.tolist(),
        prefix="validation_",
    )
    validation_metrics["validation_ece_raw"] = compute_ece(
        labels.int().tolist(), final_validation["probabilities"].tolist()
    )
    validation_metrics["validation_ece_calibrated"] = compute_ece(
        labels.int().tolist(), calibrated_probabilities.tolist()
    )
    metrics_path = Path(cfg["results"]["metrics_dir"]) / "validation_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(validation_metrics, indent=2), encoding="utf-8")

    try:
        export_model_safetensors(checkpoint_dir, best_filename)
    except Exception as exc:
        logger.warning(
            "Training is valid, but optional model_only.safetensors export failed: %s",
            exc,
        )

    return {
        "status": "completed",
        "early_stopped": stopped_early,
        "checkpoint": str(best_path),
        "calibration": calibration,
        "validation_metrics": validation_metrics,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Continue from the latest checkpoint when one exists (default).",
    )
    resume_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore existing checkpoints and train from scratch.",
    )
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, seed=args.seed)

    configure_logging(
        cfg["logging"].get("level", "INFO"),
        Path(cfg["results"]["logs_dir"]) / "training.log",
    )
    logger.info(
        "Run: experiment=%s seed=%s model=%s ablation=%s device=%s resume=%s",
        cfg["experiment"]["name"],
        cfg["training"]["seed"],
        cfg["model"]["type"],
        cfg["model"].get("ablation_mode") or "full",
        "cuda" if torch.cuda.is_available() else "cpu",
        args.resume,
    )
    with InterruptGuard() as guard:
        result = train(cfg, resume=args.resume, smoke_test=args.smoke_test, guard=guard)
    print(json.dumps(result, indent=2))
    if result.get("status") == "interrupted":
        raise SystemExit(INTERRUPTED_EXIT_CODE)


if __name__ == "__main__":
    main()
