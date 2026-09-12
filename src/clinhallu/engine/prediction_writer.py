"""Write aligned per-example predictions for audit and error analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def write_predictions(
    samples: Iterable[dict],
    labels,
    raw_probabilities,
    calibrated_probabilities,
    predictions,
    path: str | Path,
    evidence_stats: list[dict] | None = None,
) -> None:
    samples = list(samples)
    labels = list(labels)
    raw_probabilities = list(raw_probabilities)
    calibrated_probabilities = list(calibrated_probabilities)
    predictions = list(predictions)
    expected = len(samples)
    lengths = {
        "labels": len(labels),
        "raw_probabilities": len(raw_probabilities),
        "calibrated_probabilities": len(calibrated_probabilities),
        "predictions": len(predictions),
    }
    if any(length != expected for length in lengths.values()):
        raise ValueError(f"Prediction output is misaligned: samples={expected}, {lengths}")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stats = evidence_stats or [{} for _ in predictions]
    if len(stats) != expected:
        raise ValueError(
            f"Evidence statistics are misaligned: expected {expected}, got {len(stats)}"
        )
    with destination.open("w", encoding="utf-8") as handle:
        for sample, label, raw, calibrated, prediction, extra in zip(
            samples,
            labels,
            raw_probabilities,
            calibrated_probabilities,
            predictions,
            stats,
            strict=True,
        ):
            record = {
                "id": sample.get("sample_id", sample.get("id")),
                "hallu_label": int(label),
                "prediction": int(prediction),
                "p_hallucinated_raw": float(raw),
                "p_hallucinated_calibrated": float(calibrated),
            }
            record.update(extra)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
