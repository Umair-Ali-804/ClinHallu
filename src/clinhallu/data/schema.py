"""Canonical raw and processed record schemas."""

from __future__ import annotations

from typing import Any, Mapping

FIELD_ALIASES = {
    "question": ("query",),
    "answer": ("response", "generated_answer", "hallucinated_answer", "ground_truth"),
    "context": ("knowledge", "content", "evidence", "passage", "abstract"),
    "hallu_label": ("hall_label", "hallucination_label", "label"),
}


def canonical_raw_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the four text fields, ID and binary label under canonical names."""

    value = dict(record)
    for canonical, aliases in FIELD_ALIASES.items():
        if canonical not in value:
            for alias in aliases:
                if alias in value:
                    value[canonical] = value[alias]
                    break
    missing = [
        name for name in ("question", "context", "answer", "hallu_label") if name not in value
    ]
    if missing:
        raise KeyError(f"Record is missing required field(s): {', '.join(missing)}")
    label = int(value["hallu_label"])
    if label not in (0, 1):
        raise ValueError(f"hallu_label must be 0 or 1, got {label!r}")
    return {
        "id": value.get("id", value.get("sample_id", "")),
        "question": str(value["question"]).strip(),
        "context": str(value["context"]).strip(),
        "answer": str(value["answer"]).strip(),
        "hallu_label": label,
    }
