"""Build and protect immutable experiment manifests."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

_REPRODUCIBILITY_SECTIONS = (
    "model",
    "lora",
    "grounding",
    "training",
    "evaluation",
    "labels",
)


def _config_fingerprint(cfg: dict) -> str:
    """Hash settings that can change model weights or reported metrics.

    Machine-specific absolute paths are deliberately excluded. Raw file
    identity is protected separately by the split SHA-256 values.
    """

    data_keys = (
        "overlap_policy",
        "expected_rows",
        "minimum_class_fraction",
        "question_budget",
        "answer_budget",
        "context_selection",
    )
    value = {
        "experiment": cfg["experiment"],
        "sections": {name: cfg.get(name) for name in _REPRODUCIBILITY_SECTIONS},
        "data_protocol": {name: cfg["data"].get(name) for name in data_keys},
    }
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_manifest(cfg: dict, data_report: dict) -> dict:
    return {
        "protocol_version": cfg.get("project", {}).get("protocol_version", 3),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": cfg["experiment"],
        "seed": int(cfg["training"]["seed"]),
        "config_sha256": _config_fingerprint(cfg),
        "model": {
            "type": cfg["model"]["type"],
            "encoder": cfg["model"]["encoder"],
            "revision": cfg["model"].get("revision"),
            "ablation_mode": cfg["model"].get("ablation_mode"),
        },
        "data": data_report,
        "split_usage": {
            "train": "parameter optimization only",
            "validation": "checkpoint selection, temperature and threshold only",
            "test": "frozen final evaluation only",
        },
    }


def write_or_verify_manifest(path: str | Path, manifest: dict) -> None:
    destination = Path(path)
    if destination.is_file():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        if previous.get("data", {}).get("sha256") != manifest["data"]["sha256"]:
            raise RuntimeError(
                "The raw dataset changed after this run was created. Use a new "
                "experiment name or archive the previous run first."
            )
        if previous.get("config_sha256") != manifest["config_sha256"]:
            raise RuntimeError(
                "Configuration changed for an existing run directory. Use a "
                "new experiment name or archive the previous run first."
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
