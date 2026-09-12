"""Content-addressed score cache for expensive neural baselines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .data import sha256


def cached_scores(
    cache_root,
    method,
    model_name,
    revision,
    input_path,
    config,
    compute,
):
    identity = {
        "method": method,
        "model_name": model_name,
        "revision": revision,
        "input_sha256": sha256(input_path),
        "config": config,
    }
    encoded = json.dumps(identity, sort_keys=True).encode("utf-8")
    key = hashlib.sha256(encoded).hexdigest()
    directory = Path(cache_root) / method / key
    score_path = directory / "scores.npy"
    if score_path.is_file():
        return np.load(score_path, allow_pickle=False).tolist()

    values = np.asarray(compute(), dtype=np.float64)
    directory.mkdir(parents=True, exist_ok=True)
    temporary_scores = score_path.with_suffix(".npy.tmp")
    with temporary_scores.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    temporary_scores.replace(score_path)

    manifest_path = directory / "cache_manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(identity, indent=2),
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    return values.tolist()
