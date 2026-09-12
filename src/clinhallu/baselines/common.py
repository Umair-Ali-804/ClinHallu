"""Shared metrics, manifests, and writers for publication baselines."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef

from clinhallu.engine.metrics import compute_binary_metrics, compute_ece

from .data import OPTIONAL_METADATA, sha256


def select_threshold(y_true, scores, objective: str = "f1") -> tuple[float, float]:
    """Choose a threshold using validation arrays only.

    MCC is the deterministic tie-breaker, followed by the lower threshold.
    Evaluation labels never enter this function.
    """

    if objective not in {"f1", "macro_f1"}:
        raise ValueError("threshold objective must be 'f1' or 'macro_f1'")
    labels = np.asarray(y_true, dtype=int)
    values = np.asarray(scores, dtype=float)
    if len(labels) == 0 or len(labels) != len(values):
        raise ValueError("Validation labels and scores must have equal non-zero length")
    candidates = np.unique(np.r_[0.0, values, 1.0])
    best = None
    for threshold in candidates:
        predictions = (values >= threshold).astype(int)
        objective_value = f1_score(
            labels,
            predictions,
            average="macro" if objective == "macro_f1" else "binary",
            zero_division=0,
        )
        candidate = (
            float(objective_value),
            float(matthews_corrcoef(labels, predictions)),
            -float(threshold),
        )
        if best is None or candidate > best:
            best = candidate
    return -best[2], best[0]


def metrics(y_true, scores, threshold: float) -> dict[str, Any]:
    labels = np.asarray(y_true, dtype=int)
    values = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    predictions = (values >= threshold).astype(int)
    result = compute_binary_metrics(labels, predictions, values)
    result["macro_f1"] = float(f1_score(labels, predictions, average="macro", zero_division=0))
    result["ece"] = float(compute_ece(labels, values))
    result["threshold"] = float(threshold)
    return result


def prediction_rows(
    records,
    labels,
    scores,
    model: str,
    seed=None,
    score_is_probability: bool = True,
    extras: list[dict] | None = None,
) -> list[dict]:
    labels = list(labels)
    scores = list(scores)
    if len(records) != len(labels) or len(records) != len(scores):
        raise ValueError(
            "Baseline predictions are not aligned with evaluation records: "
            f"records={len(records)}, labels={len(labels)}, scores={len(scores)}"
        )
    if extras is not None and len(extras) != len(records):
        raise ValueError(f"Baseline metadata has {len(extras)} rows; expected {len(records)}")

    output = []
    for index, (row, prediction, score) in enumerate(zip(records, labels, scores, strict=True)):
        item = {
            "id": row["id"],
            "gold_label": int(row["hallu_label"]),
            "predicted_label": int(prediction),
            "model": model,
            "seed": seed,
        }
        if score_is_probability:
            item["probability_hallucinated"] = float(score)
            item["probability_supported"] = float(1.0 - score)
        else:
            item["hallucination_score"] = float(score)
        for key in OPTIONAL_METADATA:
            if key in row:
                item[key] = row[key]
        if extras is not None:
            item.update(extras[index])
        output.append(item)
    return output


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(destination)


def write_jsonl(path: str | Path, rows) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(destination)


def environment() -> dict[str, Any]:
    information = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import torch

        information.update(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            cuda_available=torch.cuda.is_available(),
            gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        )
    except ImportError:
        information["torch"] = None
    try:
        import transformers

        information["transformers"] = transformers.__version__
    except ImportError:
        information["transformers"] = None
    return information


def manifest(
    model: str,
    model_name: str,
    seed,
    paths: dict[str, str],
    config: dict,
    counts: dict[str, int],
    **extra,
) -> dict:
    value = {
        "model": model,
        "model_name": model_name,
        "seed": seed,
        "train_rows": counts["train"],
        "validation_rows": counts["val"],
        "evaluation_rows": counts["eval"],
        "train_sha256": sha256(paths["train"]),
        "val_sha256": sha256(paths["val"]),
        "eval_sha256": sha256(paths["eval"]),
        "config": config,
        "threshold_source": "validation",
        "evaluation_frozen": True,
        "environment": environment(),
    }
    value.update(extra)
    return value


def finish_run(
    output_dir,
    model,
    model_name,
    seed,
    paths,
    config,
    splits,
    validation_scores,
    evaluation_scores,
    score_is_probability=True,
    extras=None,
    threshold_objective="f1",
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train, validation, evaluation = splits
    threshold, validation_objective_value = select_threshold(
        [row["hallu_label"] for row in validation],
        validation_scores,
        threshold_objective,
    )
    evaluation_labels = [row["hallu_label"] for row in evaluation]
    result = metrics(evaluation_labels, evaluation_scores, threshold)
    if not score_is_probability:
        # Ranking metrics are valid for normalized scores; probability
        # calibration metrics are not.
        result.pop("brier", None)
        result.pop("ece", None)
    result[f"validation_{threshold_objective}_at_threshold"] = validation_objective_value
    predictions = prediction_rows(
        evaluation,
        np.asarray(evaluation_scores) >= threshold,
        evaluation_scores,
        model,
        seed,
        score_is_probability,
        extras,
    )
    write_jsonl(output / "predictions.jsonl", predictions)
    write_json(output / "metrics.json", result)
    paths_by_split = {"train": paths[0], "val": paths[1], "eval": paths[2]}
    run_manifest = manifest(
        model,
        model_name,
        seed,
        paths_by_split,
        config,
        {
            "train": len(train),
            "val": len(validation),
            "eval": len(evaluation),
        },
        threshold_objective=threshold_objective,
        validation_objective_value=validation_objective_value,
        score_semantics=("probability" if score_is_probability else "normalized_score"),
    )
    write_json(output / "run_manifest.json", run_manifest)
    write_json(output / "status.json", {"status": "completed", "model": model})
    return result


def finish_fixed_prediction_run(
    output_dir,
    model,
    model_name,
    paths,
    splits,
    evaluation_labels,
    extras=None,
):
    """Evaluate externally supplied binary judge predictions at threshold 0.5."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train, validation, evaluation = splits
    scores = np.asarray(evaluation_labels, dtype=float)
    gold = [row["hallu_label"] for row in evaluation]
    result = metrics(gold, scores, 0.5)
    result.pop("brier", None)
    result.pop("ece", None)
    result["score_semantics"] = "binary_label_only"
    predictions = prediction_rows(
        evaluation,
        scores.astype(int),
        scores,
        model,
        None,
        False,
        extras,
    )
    write_jsonl(output / "predictions.jsonl", predictions)
    write_json(output / "metrics.json", result)
    paths_by_split = {"train": paths[0], "val": paths[1], "eval": paths[2]}
    run_manifest = manifest(
        model,
        model_name,
        None,
        paths_by_split,
        {"decision_rule": "externally supplied binary label"},
        {
            "train": len(train),
            "val": len(validation),
            "eval": len(evaluation),
        },
        threshold_source="fixed_binary_output",
        score_semantics="binary_label_only",
    )
    write_json(output / "run_manifest.json", run_manifest)
    write_json(output / "status.json", {"status": "completed", "model": model})
    return result
