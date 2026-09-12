"""Durable checkpoint persistence with epoch- and step-level resume.

Three properties matter when a run can die at any moment (Colab disconnect,
closed laptop, power cut):

``durable``
    The bytes are on the platter before the file becomes visible. Every write
    goes to a temporary file that is ``fsync``-ed, then atomically renamed, then
    the directory itself is ``fsync``-ed. A crash therefore leaves either the
    previous checkpoint or the new one -- never a half-written file.

``redundant``
    The previous ``latest`` checkpoint is rotated to ``*.bak`` before being
    replaced. If the newest file is somehow unreadable (a filesystem that lies
    about ``fsync``, a truncated Google Drive sync), the run falls back one
    checkpoint instead of restarting from epoch 1.

``complete``
    Model, optimizer, scheduler and AMP-scaler state are stored together with
    the position inside the epoch, the RNG state of every library that affects
    results, and the partially accumulated metrics for the in-flight epoch. A
    resumed run therefore continues the same trajectory rather than a similar
    one.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

#: Bumped when the on-disk layout changes in a way older readers cannot handle.
CHECKPOINT_FORMAT_VERSION = 2

#: Written after a training pass finishes but before validation has been scored.
PHASE_EPOCH_TRAINED = "epoch_trained"
#: Written mid-epoch, at an optimizer-step boundary.
PHASE_IN_EPOCH = "in_epoch"
#: Written once an epoch is fully scored and selection has been applied.
PHASE_EPOCH_COMPLETE = "epoch_complete"


@dataclass
class CheckpointState:
    """Where a resumed run should pick up.

    ``start_epoch`` is the epoch to begin (or continue) with. When
    ``batches_done`` is non-zero the run resumes *inside* that epoch, having
    already consumed ``samples_done`` examples from its permutation.
    """

    start_epoch: int = 1
    global_step: int = 0
    best_metric: float = -1.0
    patience_count: int = 0
    resumed: bool = False
    phase: str = PHASE_EPOCH_COMPLETE
    batches_done: int = 0
    samples_done: int = 0
    epoch_progress: Dict[str, Any] = field(default_factory=dict)
    history: List[dict] = field(default_factory=list)

    @property
    def mid_epoch(self) -> bool:
        return self.phase == PHASE_IN_EPOCH and self.batches_done > 0

    @property
    def needs_validation_only(self) -> bool:
        """Training for ``start_epoch`` finished; only scoring is outstanding."""
        return self.phase == PHASE_EPOCH_TRAINED


def capture_rng_state() -> Dict[str, Any]:
    """Snapshot every RNG that can change training results.

    Values are reduced to plain Python types and tensors so the checkpoint stays
    loadable under ``torch.load(weights_only=True)``.
    """

    numpy_state = np.random.get_state()
    python_state = random.getstate()
    payload: Dict[str, Any] = {
        "python": [int(value) for value in python_state[1]],
        "python_pos": int(python_state[2] if python_state[2] is not None else 624),
        "numpy_keys": [int(value) for value in numpy_state[1]],
        "numpy_pos": int(numpy_state[2]),
        "numpy_has_gauss": int(numpy_state[3]),
        "numpy_cached_gauss": float(numpy_state[4]),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = torch.cuda.get_rng_state_all()
    return payload


def restore_rng_state(payload: Optional[Dict[str, Any]]) -> None:
    """Reinstate a snapshot from :func:`capture_rng_state`, tolerating gaps."""

    if not payload:
        return
    try:
        keys = tuple(int(value) for value in payload["python"])
        random.setstate((3, keys, int(payload.get("python_pos", 624))))
    except Exception:  # pragma: no cover - defensive
        logger.debug("Could not restore the Python RNG state")
    try:
        np.random.set_state(
            (
                "MT19937",
                np.array(payload["numpy_keys"], dtype=np.uint32),
                int(payload["numpy_pos"]),
                int(payload["numpy_has_gauss"]),
                float(payload["numpy_cached_gauss"]),
            )
        )
    except Exception:  # pragma: no cover - defensive
        logger.debug("Could not restore the NumPy RNG state")
    try:
        torch.set_rng_state(payload["torch"].cpu().to(torch.uint8))
    except Exception:  # pragma: no cover - defensive
        logger.debug("Could not restore the PyTorch RNG state")
    cuda_state = payload.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([tensor.cpu().to(torch.uint8) for tensor in cuda_state])
        except Exception:  # pragma: no cover - defensive
            logger.debug("Could not restore the CUDA RNG state")


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":  # Windows has no directory file descriptors.
        return
    try:
        handle = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(handle)
    except OSError:
        pass
    finally:
        os.close(handle)


def atomic_torch_save(payload: Dict[str, Any], target: Path) -> Path:
    """Write ``payload`` so the destination is either old or new, never partial."""

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)
    _fsync_directory(target.parent)
    return target


def save_checkpoint(
    checkpoint_dir: str,
    filename: str,
    epoch: int,
    global_step: int,
    model_state: Dict[str, Any],
    optimizer_state: Dict[str, Any],
    scheduler_state: Dict[str, Any],
    scaler_state: Optional[Dict[str, Any]],
    best_metric: float,
    patience_count: int,
    extra: Optional[Dict] = None,
    *,
    phase: str = PHASE_EPOCH_COMPLETE,
    batches_done: int = 0,
    samples_done: int = 0,
    epoch_progress: Optional[Dict[str, Any]] = None,
    history: Optional[List[dict]] = None,
    rng_state: Optional[Dict[str, Any]] = None,
    keep_backup: bool = True,
) -> Path:
    """Persist full training state and return the checkpoint path."""

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "epoch": epoch,
        "global_step": global_step,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "scaler_state": scaler_state,
        "best_metric": best_metric,
        "patience_count": patience_count,
        "phase": phase,
        "batches_done": int(batches_done),
        "samples_done": int(samples_done),
        "epoch_progress": epoch_progress or {},
        "history": history or [],
        "rng_state": rng_state if rng_state is not None else capture_rng_state(),
    }
    if extra:
        payload["extra"] = extra

    target = ckpt_dir / filename
    if keep_backup and target.is_file():
        backup = ckpt_dir / (filename + ".bak")
        try:
            os.replace(str(target), str(backup))
        except OSError:  # pragma: no cover - best effort only
            logger.debug("Could not rotate %s to a backup", target)

    atomic_torch_save(payload, target)
    logger.debug(
        "Checkpoint saved: %s (epoch=%d, step=%d, phase=%s, batches_done=%d)",
        target,
        epoch,
        global_step,
        phase,
        batches_done,
    )
    return target


def export_model_only_checkpoint(
    checkpoint_dir: str,
    src_filename: str,
    dst_filename: str = "model_only.pt",
    extra: Optional[Dict] = None,
) -> Path:
    """Copy weights and metadata from a full checkpoint into a smaller file."""
    src = Path(checkpoint_dir) / src_filename
    payload = load_checkpoint(str(src), device=torch.device("cpu"))

    model_only_payload = {
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "model_state": payload["model_state"],
        "best_metric": payload.get("best_metric"),
    }
    if payload.get("extra"):
        model_only_payload["extra"] = payload["extra"]
    if extra:
        model_only_payload.setdefault("extra", {}).update(extra)

    target = Path(checkpoint_dir) / dst_filename
    atomic_torch_save(model_only_payload, target)

    logger.info(
        "Model-only checkpoint exported: %s (%.1f MB) from %s (%.1f MB) -- "
        "no optimizer/scheduler/scaler state included.",
        target,
        target.stat().st_size / 1e6,
        src,
        src.stat().st_size / 1e6,
    )
    return target


def export_model_safetensors(
    checkpoint_dir: str,
    src_filename: str,
    dst_filename: str = "model_only.safetensors",
) -> Path:
    """Export only tensor weights in the portable safetensors format."""

    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ImportError("Install safetensors before exporting model_only.safetensors") from exc

    source = Path(checkpoint_dir) / src_filename
    payload = load_checkpoint(str(source), torch.device("cpu"))
    # Clone tensors so tied/shared storage cannot make safetensors reject the
    # export. The regular best_model.pt remains the authoritative resumable
    # checkpoint and contains metadata.
    state = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in payload["model_state"].items()
    }
    destination = Path(checkpoint_dir) / dst_filename
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    save_file(state, str(temporary), metadata={"format": "pt"})
    temporary.replace(destination)
    _fsync_directory(destination.parent)
    logger.info("Model-only safetensors exported to %s", destination)
    return destination


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        payload = torch.load(str(path), map_location=device, weights_only=True)
    except Exception:
        logger.warning(
            "weights_only=True failed for %s (likely legacy checkpoint with "
            "non-tensor objects in 'extra').  Retrying with weights_only=False. "
            "Re-save with the current code to eliminate this warning.",
            path,
        )
        payload = torch.load(str(path), map_location=device, weights_only=False)

    logger.info("Loaded checkpoint from %s (epoch=%s)", path, payload.get("epoch", "?"))
    return payload


def _resume_candidates(checkpoint_dir: str, filename: str) -> List[Path]:
    """Files that could carry resume state, newest first."""
    ckpt_dir = Path(checkpoint_dir)
    return [
        path for path in (ckpt_dir / filename, ckpt_dir / (filename + ".bak")) if path.is_file()
    ]


def find_latest_checkpoint(
    checkpoint_dir: str,
    filename: str,
    device: torch.device,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    *,
    restore_rng: bool = True,
) -> CheckpointState:
    """Load the newest usable checkpoint and restore every component from it."""

    candidates = _resume_candidates(checkpoint_dir, filename)
    if not candidates:
        logger.info("No checkpoint at %s — starting from scratch.", Path(checkpoint_dir) / filename)
        return CheckpointState()

    payload = None
    errors = []
    for candidate in candidates:
        try:
            payload = load_checkpoint(str(candidate), device)
            if candidate.name.endswith(".bak"):
                logger.warning(
                    "The newest checkpoint was unreadable; resuming from the backup %s. "
                    "At most one checkpoint interval of progress was lost.",
                    candidate,
                )
            break
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
            logger.warning("Checkpoint %s could not be read (%s); trying a backup.", candidate, exc)

    if payload is None:
        raise RuntimeError(
            "Every resume checkpoint is unreadable, so continuing would silently "
            "discard prior work. Inspect or delete these files and re-run:\n  "
            + "\n  ".join(errors)
        )

    ckpt_keys = set(payload["model_state"].keys())
    model_keys = set(model.state_dict().keys())

    overlap = ckpt_keys & model_keys
    if len(overlap) < max(len(ckpt_keys), len(model_keys)) * 0.5:
        raise RuntimeError(
            "Checkpoint appears incompatible with the current model "
            f"architecture: only {len(overlap)}/{max(len(ckpt_keys), len(model_keys))} "
            f"state keys match in {checkpoint_dir}/{filename}. Use a new run directory "
            "or the checkpoint migration script."
        )

    try:
        model.load_state_dict(payload["model_state"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint state does not exactly match the configured model: "
            f"{checkpoint_dir}/{filename}"
        ) from exc
    if optimizer and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler and payload.get("scheduler_state"):
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])
    if restore_rng:
        restore_rng_state(payload.get("rng_state"))

    phase = payload.get("phase", PHASE_EPOCH_COMPLETE)
    epoch = int(payload["epoch"])
    # Only a fully scored epoch advances the counter; the other phases continue
    # working on the epoch that was in flight.
    start_epoch = epoch + 1 if phase == PHASE_EPOCH_COMPLETE else epoch

    state = CheckpointState(
        start_epoch=start_epoch,
        global_step=int(payload.get("global_step", 0)),
        best_metric=float(payload.get("best_metric", -1.0)),
        patience_count=int(payload.get("patience_count", 0)),
        resumed=True,
        phase=phase,
        batches_done=int(payload.get("batches_done", 0)),
        samples_done=int(payload.get("samples_done", 0)),
        epoch_progress=payload.get("epoch_progress") or {},
        history=list(payload.get("history") or []),
    )
    if state.mid_epoch:
        logger.info(
            "Resuming inside epoch %d after %d batches (%d examples, step=%d, best=%.4f)",
            state.start_epoch,
            state.batches_done,
            state.samples_done,
            state.global_step,
            state.best_metric,
        )
    elif state.needs_validation_only:
        logger.info(
            "Epoch %d training already finished; resuming at its validation pass.",
            state.start_epoch,
        )
    else:
        logger.info(
            "Resumed from epoch %d (step=%d, best=%.4f)",
            state.start_epoch,
            state.global_step,
            state.best_metric,
        )
    return state
