"""Verify completed runs and fixed validation/evaluation identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REQUIRED_METRICS = (
    "eval_f1",
    "eval_mcc",
    "eval_auc",
    "eval_auprc",
    "eval_brier",
    "eval_confusion_matrix",
    "temperature",
    "threshold",
)


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _load_predictions(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def verify_run(run_dir: str | Path) -> dict:
    run = Path(run_dir).resolve()
    files = {
        "config": run / "resolved_config.yaml",
        "manifest": run / "run_manifest.json",
        "state": run / "run_state.json",
        "environment": run / "environment.json",
        "checkpoint": run / "checkpoints" / "best_model.pt",
        "temperature": run / "calibration" / "temperature.npy",
        "threshold": run / "calibration" / "threshold.npy",
        "metrics": run / "metrics" / "evaluation_metrics.json",
        "predictions": run / "predictions" / "evaluation_predictions.jsonl",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing run artifact(s): " + ", ".join(missing))

    manifest = _load_json(files["manifest"])
    state = _load_json(files["state"])
    metrics = _load_json(files["metrics"])
    missing_metrics = [key for key in REQUIRED_METRICS if key not in metrics]
    if missing_metrics:
        raise KeyError("Missing metric(s): " + ", ".join(missing_metrics))
    if metrics.get("protocol") != "frozen_test_after_validation_calibration":
        raise ValueError(f"Unexpected evaluation protocol in {files['metrics']}")

    temperature = float(np.load(files["temperature"]).reshape(-1)[0])
    threshold = float(np.load(files["threshold"]).reshape(-1)[0])
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"Invalid temperature in {files['temperature']}")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(f"Invalid threshold in {files['threshold']}")
    if not np.isclose(float(metrics["temperature"]), temperature):
        raise ValueError("Metric and calibration temperature values differ")
    if not np.isclose(float(metrics["threshold"]), threshold):
        raise ValueError("Metric and calibration threshold values differ")

    predictions = _load_predictions(files["predictions"])
    expected_count = manifest.get("data", {}).get("counts", {}).get("test")
    if expected_count is not None and len(predictions) != int(expected_count):
        raise ValueError(f"Expected {expected_count} predictions, observed {len(predictions)}")
    identifiers = [str(row.get("id", "")) for row in predictions]
    if any(not identifier for identifier in identifiers):
        raise ValueError("Every prediction must have a non-empty id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Prediction ids are not unique")
    if state.get("stage") != "evaluated":
        raise ValueError(f"Run state is {state.get('stage')!r}, expected 'evaluated'")

    hashes = manifest.get("data", {}).get("sha256", {})
    return {
        "status": "valid",
        "run_dir": str(run),
        "prediction_count": len(predictions),
        "validation_sha256": hashes.get("validation"),
        "evaluation_sha256": hashes.get("test"),
        "metrics": {key: metrics[key] for key in REQUIRED_METRICS},
    }


def _verify_shared_frozen_splits(results: list[dict]) -> None:
    for key, label in (
        ("validation_sha256", "validation"),
        ("evaluation_sha256", "evaluation"),
    ):
        hashes = {result[key] for result in results}
        if None in hashes:
            raise KeyError(f"At least one manifest has no {label} SHA-256")
        if len(hashes) != 1:
            raise RuntimeError(f"Runs did not use identical {label} files: {sorted(hashes)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args()
    results = [verify_run(path) for path in args.run_dirs]
    if len(results) > 1:
        _verify_shared_frozen_splits(results)
    print(
        json.dumps(
            {
                "status": "valid",
                "runs": results,
                "shared_frozen_splits": len(results) > 1,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
