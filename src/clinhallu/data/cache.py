"""Content-addressed, compact tokenization caches for processed splits."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import torch
from torch.utils.data import Dataset

from .context_selection import context_selection_from_config
from .dataset import HKGFusionDataset
from .io import read_jsonl, sha256_file

logger = logging.getLogger(__name__)
FEATURE_SCHEMA_VERSION = 1
SPLIT_FILENAMES = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "test": "test.jsonl",
}


def cache_key(payload: Mapping) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def write_cache_manifest(path: str | Path, payload: Mapping) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(destination)


def _cache_identity(cfg: Mapping, tokenizer) -> dict:
    processed_dir = Path(cfg["data"]["processed_dir"])
    split_hashes = {
        split: sha256_file(processed_dir / filename) for split, filename in SPLIT_FILENAMES.items()
    }
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "split_sha256": split_hashes,
        "tokenizer": {
            "class": tokenizer.__class__.__name__,
            "name_or_path": str(getattr(tokenizer, "name_or_path", "unknown")),
            "revision": cfg["model"].get("revision"),
            "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
            "cls_token_id": getattr(tokenizer, "cls_token_id", None),
            "sep_token_id": getattr(tokenizer, "sep_token_id", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        },
        "encoding": {
            "max_length": int(cfg["model"]["max_length"]),
            "question_budget": int(cfg["data"]["question_budget"]),
            "answer_budget": int(cfg["data"]["answer_budget"]),
            "context_selection": cfg["data"].get("context_selection", {}),
        },
    }


def feature_cache_directory(cfg: Mapping, tokenizer) -> tuple[Path, dict]:
    identity = _cache_identity(cfg, tokenizer)
    return Path(cfg["data"]["cache_dir"]) / cache_key(identity), identity


class CachedFeatureDataset(Dataset):
    """Dataset view over compact, fixed-length CPU feature tensors."""

    def __init__(self, payload: dict, samples: list[dict]) -> None:
        self.features = payload["features"]
        self.samples = samples
        self.max_length = int(payload["max_length"])
        if len(self.samples) != int(payload["count"]):
            raise ValueError("Cached feature count does not match processed records")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        length = int(self.features["lengths"][index].item())
        item = {
            "input_ids": self.features["input_ids"][index, :length].long(),
            "attention_mask": self.features["attention_mask"][index, :length].long(),
            "question_mask": self.features["question_mask"][index, :length].float(),
            "context_mask": self.features["context_mask"][index, :length].float(),
            "answer_mask": self.features["answer_mask"][index, :length].float(),
        }
        if "labels" in self.features:
            item["hall_label"] = self.features["labels"][index].long()
        return item


def _load_payload(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _valid_payload(payload: dict, source_hash: str, count: int, max_length: int) -> bool:
    return (
        payload.get("feature_schema_version") == FEATURE_SCHEMA_VERSION
        and payload.get("source_sha256") == source_hash
        and int(payload.get("count", -1)) == count
        and int(payload.get("max_length", -1)) == max_length
        and isinstance(payload.get("features"), dict)
    )


def _build_payload(cfg: Mapping, tokenizer, source_path: Path) -> tuple[dict, list[dict]]:
    selector = context_selection_from_config(cfg)
    source = HKGFusionDataset(
        str(source_path),
        tokenizer,
        max_length=int(cfg["model"]["max_length"]),
        context_selection=selector,
        question_budget=int(cfg["data"]["question_budget"]),
        answer_budget=int(cfg["data"]["answer_budget"]),
    )
    count = len(source)
    max_length = int(cfg["model"]["max_length"])
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0

    features = {
        "input_ids": torch.full((count, max_length), int(pad_token_id), dtype=torch.int32),
        "attention_mask": torch.zeros((count, max_length), dtype=torch.bool),
        "question_mask": torch.zeros((count, max_length), dtype=torch.bool),
        "context_mask": torch.zeros((count, max_length), dtype=torch.bool),
        "answer_mask": torch.zeros((count, max_length), dtype=torch.bool),
        "labels": torch.empty(count, dtype=torch.int8),
        "lengths": torch.empty(count, dtype=torch.int32),
    }
    for index in range(count):
        item = source[index]
        length = int(item["input_ids"].numel())
        features["lengths"][index] = length
        features["input_ids"][index, :length] = item["input_ids"].to(torch.int32)
        for name in (
            "attention_mask",
            "question_mask",
            "context_mask",
            "answer_mask",
        ):
            features[name][index, :length] = item[name].bool()
        features["labels"][index] = item["hall_label"].to(torch.int8)

    payload = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "source_sha256": sha256_file(source_path),
        "count": count,
        "max_length": max_length,
        "features": features,
    }
    return payload, source.samples


def load_or_build_feature_dataset(
    cfg: Mapping,
    tokenizer,
    split: str,
) -> Dataset:
    """Load one valid cache file, or atomically rebuild it when stale."""

    if split not in SPLIT_FILENAMES:
        raise ValueError(f"Unknown split {split!r}; choose from {sorted(SPLIT_FILENAMES)}")
    source_path = Path(cfg["data"]["processed_dir"]) / SPLIT_FILENAMES[split]
    if not source_path.is_file():
        raise FileNotFoundError(f"Processed split not found: {source_path}")

    cache_dir, identity = feature_cache_directory(cfg, tokenizer)
    target = cache_dir / f"{split}_features.pt"
    source_hash = identity["split_sha256"][split]
    samples = read_jsonl(source_path)
    max_length = int(cfg["model"]["max_length"])
    payload = None
    if target.is_file():
        try:
            candidate = _load_payload(target)
            if _valid_payload(candidate, source_hash, len(samples), max_length):
                payload = candidate
                logger.info("Using tokenization cache: %s", target)
        except Exception as exc:
            logger.warning("Ignoring invalid tokenization cache %s: %s", target, exc)

    if payload is None:
        logger.info("Building tokenization cache for %s", source_path)
        payload, samples = _build_payload(cfg, tokenizer, source_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(target)

    manifest_path = cache_dir / "cache_manifest.json"
    manifest = {
        "cache_key": cache_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "files": {
            name: str(cache_dir / f"{name}_features.pt")
            for name in SPLIT_FILENAMES
            if (cache_dir / f"{name}_features.pt").is_file()
        },
    }
    write_cache_manifest(manifest_path, manifest)
    return CachedFeatureDataset(payload, samples)
