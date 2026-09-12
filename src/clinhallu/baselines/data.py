"""Strict fixed-split loading shared by all publication baselines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = ("id", "question", "context", "answer", "hallu_label")
OPTIONAL_METADATA = (
    "group_id",
    "source_dataset",
    "difficulty",
    "generation_model",
    "annotation_type",
    "reason",
    "category",
    "hallucination_category",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(path: str | Path, require_labels: bool = True) -> list[dict[str, Any]]:
    records, ids = [], set()
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            if type(row.get("id")) not in (str, int):
                raise ValueError(f"{path}:{line_no}: id must be string or integer")
            for key in ("question", "context", "answer"):
                if not isinstance(row.get(key), str):
                    raise ValueError(f"{path}:{line_no}: {key} must be a string")
            required = REQUIRED_FIELDS if require_labels else REQUIRED_FIELDS[:-1]
            missing = [key for key in required if key not in row]
            if missing:
                raise ValueError(f"{path}:{line_no}: missing fields {missing}")
            for key in ("id", "question", "context", "answer"):
                if not str(row.get(key, "")).strip():
                    raise ValueError(f"{path}:{line_no}: {key!r} is empty")
            identifier = str(row["id"])
            if identifier in ids:
                raise ValueError(f"{path}:{line_no}: duplicate id {row['id']!r}")
            ids.add(identifier)
            if require_labels:
                label = row["hallu_label"]
                if isinstance(label, bool) or label not in (0, 1, 0.0, 1.0):
                    raise ValueError(f"{path}:{line_no}: hallu_label must be integer 0 or 1")
                row["hallu_label"] = int(label)
            records.append(row)
    if not records:
        raise ValueError(f"{path}: no records")
    if require_labels and set(r["hallu_label"] for r in records) != {0, 1}:
        raise ValueError(f"{path}: both labels 0 and 1 are required")
    return records


def serialize(row: dict[str, Any]) -> str:
    return f"Question: {row['question']}\n\nContext: {row['context']}\n\nAnswer: {row['answer']}"


def sentence_split(text: str, maximum: int | None = None) -> list[str]:
    import re

    pieces = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text) if p.strip()]
    pieces = pieces or [text.strip()]
    return pieces[:maximum] if maximum else pieces


def assert_fixed_protocol(train: list[dict], val: list[dict], evaluation: list[dict]) -> None:
    def group(row: dict) -> str:
        if row.get("group_id") not in (None, ""):
            return "group:" + str(row["group_id"])
        normalized = " ".join((str(row["question"]) + " " + str(row["context"])).lower().split())
        return "text:" + hashlib.sha256(normalized.encode()).hexdigest()

    groups = {
        name: {group(r) for r in rows}
        for name, rows in (("train", train), ("val", val), ("eval", evaluation))
    }
    for left, right in (("train", "val"), ("train", "eval"), ("val", "eval")):
        overlap = groups[left] & groups[right]
        if overlap:
            raise ValueError(f"Data leakage: {len(overlap)} shared groups in {left}/{right}")


def load_splits(train_path: str, val_path: str, eval_path: str):
    train, val, evaluation = (load_records(p) for p in (train_path, val_path, eval_path))
    assert_fixed_protocol(train, val, evaluation)
    return train, val, evaluation
