"""Single idempotent runner for preparation, training and frozen evaluation.

Every stage is recorded in a ledger inside the run directory as it starts and
finishes, so re-invoking the same command after a crash walks straight past the
stages that already completed and picks up the first one that did not. Training
itself resumes from its own checkpoint, so even a stage that was half-finished
continues rather than restarting.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

from clinhallu.config import load_config, resolve_config, save_resolved_config
from clinhallu.core.environment import write_environment
from clinhallu.core.interrupt import INTERRUPTED_EXIT_CODE, InterruptedRun
from clinhallu.core.logging import configure_logging
from clinhallu.data.io import sha256_file
from clinhallu.data.preprocessing import run_fixed_preprocessing
from clinhallu.data.validation import validate_fixed_splits

from .ledger import Ledger, fingerprint
from .manifest import build_manifest, write_or_verify_manifest
from .state import RunState

logger = logging.getLogger(__name__)

STAGE_PREPARE = "prepare"
STAGE_TRAIN = "train"
STAGE_EVALUATE = "evaluate"


def _preprocessing_signature(cfg: dict, data_report: dict) -> str:
    payload = {
        "preprocessing_version": 1,
        "sha256": data_report["sha256"],
        "overlap_policy": cfg["data"].get("overlap_policy", "error"),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _processed_is_current(cfg: dict, signature: str) -> bool:
    root = Path(cfg["data"]["processed_dir"])
    required = [root / name for name in ("train.jsonl", "validation.jsonl", "test.jsonl")]
    manifest = root / "preprocessing_manifest.json"
    if not manifest.is_file() or not all(path.is_file() for path in required):
        return False
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        logger.warning("Rebuilding invalid preprocessing manifest: %s", manifest)
        return False
    observed_hashes = {path.name: sha256_file(path) for path in required}
    return (
        value.get("preprocessing_signature") == signature
        and value.get("processed_sha256") == observed_hashes
    )


def _stamp_preprocessing_manifest(cfg: dict, signature: str, data_report: dict) -> None:
    path = Path(cfg["data"]["processed_dir"]) / "preprocessing_manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["preprocessing_signature"] = signature
    value["raw_sha256"] = data_report["sha256"]
    root = Path(cfg["data"]["processed_dir"])
    value["processed_sha256"] = {
        filename: sha256_file(root / filename)
        for filename in ("train.jsonl", "validation.jsonl", "test.jsonl")
    }
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _run(command: list[str], cwd: str | Path, *, stage: str) -> None:
    """Execute a child stage, distinguishing a clean stop from a real failure."""
    logger.info("Executing: %s", " ".join(command))
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode == 0:
        return
    if completed.returncode in (INTERRUPTED_EXIT_CODE, -2):
        raise InterruptedRun(
            f"The {stage} stage stopped cleanly and saved its progress. "
            "Re-run the same command to continue from that point."
        )
    raise subprocess.CalledProcessError(completed.returncode, command)


def _run_fingerprint(cfg: dict) -> str:
    """Identify the configuration a stage result belongs to."""
    return fingerprint(
        {
            "experiment": cfg["experiment"],
            "model": cfg["model"],
            "training": cfg["training"],
            "data": {
                key: cfg["data"].get(key)
                for key in ("processed_dir", "question_budget", "answer_budget")
            },
        }
    )


def run_experiment(
    config_path: str | Path,
    *,
    seed: int | None = None,
    resume: bool = True,
    prepare_only: bool = False,
    force_evaluation: bool = False,
    smoke_test: bool = False,
    force_stages: set[str] | None = None,
) -> dict:
    cfg = load_config(config_path, seed=seed)
    if smoke_test:
        cfg["experiment"]["name"] = cfg["experiment"]["name"] + "_smoke"
        cfg = resolve_config(
            cfg,
            project_root=cfg["project_root"],
            config_path=cfg["config_path"],
        )

    force_stages = set(force_stages or ())
    if force_evaluation:
        force_stages.add(STAGE_EVALUATE)

    run_dir = Path(cfg["output"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(cfg["logging"].get("level", "INFO"), run_dir / "logs" / "run.log")
    state = RunState(run_dir / "run_state.json")
    ledger = Ledger(run_dir / "stage_ledger.json", name=cfg["experiment"]["name"])
    if not resume:
        logger.info("--no-resume: discarding recorded stage results for this run.")
        ledger.reset()

    state.update("validating")
    expected = cfg["data"].get("expected_rows", {})
    data_report = validate_fixed_splits(
        cfg["data"]["train_path"],
        cfg["data"]["val_path"],
        cfg["data"]["eval_path"],
        expected_rows=expected,
        minimum_class_fraction=float(cfg["data"].get("minimum_class_fraction", 0.0)),
    )
    dataset_name = Path(cfg["data"]["processed_dir"]).name
    data_manifest_path = Path(cfg["project_root"]) / "data" / "manifests" / f"{dataset_name}.json"
    if data_manifest_path.is_file():
        previous_data = json.loads(data_manifest_path.read_text(encoding="utf-8"))
        if previous_data.get("sha256") != data_report["sha256"]:
            raise RuntimeError(
                f"Data registered as {dataset_name!r} changed. Use a new condition name "
                "or archive the earlier processed data first."
            )
    else:
        data_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        data_manifest_path.write_text(json.dumps(data_report, indent=2), encoding="utf-8")
    manifest = build_manifest(cfg, data_report)
    write_or_verify_manifest(run_dir / "run_manifest.json", manifest)
    save_resolved_config(cfg, run_dir / "resolved_config.yaml")
    write_environment(run_dir / "environment.json")

    run_signature = _run_fingerprint(cfg)
    signature = _preprocessing_signature(cfg, data_report)

    if _processed_is_current(cfg, signature):
        logger.info("Reusing verified processed data at %s", cfg["data"]["processed_dir"])
        ledger.complete(STAGE_PREPARE, fingerprint=signature, reused=True)
    else:
        state.update("preprocessing")
        ledger.start(STAGE_PREPARE, fingerprint_value=signature)
        try:
            run_fixed_preprocessing(
                str(run_dir / "resolved_config.yaml"),
                cfg["data"]["train_path"],
                cfg["data"]["val_path"],
                cfg["data"]["eval_path"],
                overlap_policy=cfg["data"].get("overlap_policy", "error"),
            )
            _stamp_preprocessing_manifest(cfg, signature, data_report)
        except InterruptedRun:
            ledger.interrupt(STAGE_PREPARE)
            raise
        except Exception as exc:
            ledger.fail(STAGE_PREPARE, str(exc))
            raise
        ledger.complete(STAGE_PREPARE, fingerprint=signature)
    state.update("prepared")

    if prepare_only:
        return {"status": "prepared", "run_dir": str(run_dir), "data": data_report}

    resolved_config = str(run_dir / "resolved_config.yaml")
    best_checkpoint = Path(cfg["checkpoints"]["dir"]) / cfg["checkpoints"]["best_filename"]
    latest_checkpoint = Path(cfg["checkpoints"]["dir"]) / cfg["checkpoints"]["latest_filename"]
    calibration_files = (
        Path(cfg["calibration"]["dir"]) / cfg["calibration"]["temperature_filename"],
        Path(cfg["calibration"]["dir"]) / cfg["calibration"]["threshold_filename"],
    )

    artifacts_present = best_checkpoint.is_file() and all(
        path.is_file() for path in calibration_files
    )
    training_ready = artifacts_present and (
        ledger.is_complete(STAGE_TRAIN, fingerprint_value=run_signature)
        or not ledger.entry(STAGE_TRAIN)
    )
    if STAGE_TRAIN in force_stages:
        training_ready = False

    if training_ready:
        logger.info("Training artifacts already present; skipping to evaluation.")
        ledger.complete(STAGE_TRAIN, fingerprint=run_signature, checkpoint=str(best_checkpoint))
    else:
        if best_checkpoint.is_file() and not latest_checkpoint.is_file() and not artifacts_present:
            missing = [str(path) for path in calibration_files if not path.is_file()]
            raise RuntimeError(
                "A best checkpoint exists, but validation calibration is incomplete "
                "and no resumable latest checkpoint is available. Missing: " + ", ".join(missing)
            )
        state.update("training")
        ledger.start(STAGE_TRAIN, fingerprint_value=run_signature)
        command = [
            sys.executable,
            "-m",
            "clinhallu.cli.train",
            "--config",
            resolved_config,
        ]
        if smoke_test:
            command.append("--smoke-test")
        command.append("--resume" if resume else "--no-resume")
        try:
            _run(command, cfg["project_root"], stage="training")
        except InterruptedRun:
            ledger.interrupt(STAGE_TRAIN, checkpoint=str(latest_checkpoint))
            state.update("interrupted", stage="training")
            raise
        except Exception as exc:
            ledger.fail(STAGE_TRAIN, str(exc))
            raise
        ledger.complete(STAGE_TRAIN, fingerprint=run_signature, checkpoint=str(best_checkpoint))

    if not best_checkpoint.is_file() or not all(path.is_file() for path in calibration_files):
        raise RuntimeError(
            "Training did not produce the required best checkpoint and "
            "validation-calibration artifacts. See logs/training.log."
        )
    state.update("trained", checkpoint=str(best_checkpoint))

    if smoke_test:
        state.update("smoke_test_complete")
        return {"status": "smoke_test_complete", "run_dir": str(run_dir)}

    metrics_path = Path(cfg["results"]["metrics_dir"]) / "evaluation_metrics.json"
    evaluation_done = metrics_path.is_file() and ledger.is_complete(
        STAGE_EVALUATE, fingerprint_value=run_signature
    )
    if STAGE_EVALUATE in force_stages or not (evaluation_done or metrics_path.is_file()):
        state.update("evaluating")
        ledger.start(STAGE_EVALUATE, fingerprint_value=run_signature)
        command = [
            sys.executable,
            "-m",
            "clinhallu.cli.evaluate",
            "--config",
            resolved_config,
        ]
        if STAGE_EVALUATE in force_stages:
            command.append("--force")
        try:
            _run(command, cfg["project_root"], stage="evaluation")
        except InterruptedRun:
            ledger.interrupt(STAGE_EVALUATE)
            state.update("interrupted", stage="evaluation")
            raise
        except Exception as exc:
            ledger.fail(STAGE_EVALUATE, str(exc))
            raise
    ledger.complete(STAGE_EVALUATE, fingerprint=run_signature, metrics=str(metrics_path))
    state.update("evaluated", metrics=str(metrics_path))
    return {
        "status": "evaluated",
        "run_dir": str(run_dir),
        "checkpoint": str(best_checkpoint),
        "metrics": str(metrics_path),
    }
