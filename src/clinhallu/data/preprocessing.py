"""Validate and normalize user-provided fixed data splits without resampling."""

import argparse
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from clinhallu.config import load_config as load_project_config

from .schema import FIELD_ALIASES

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"question", "answer", "context", "hallu_label"}
OPTIONAL_COLUMNS = {
    "reason",
    "nli_label",
    "source_dataset",
    "source_split",
    "difficulty",
    "generation_model",
    "annotation_type",
    "group_id",
}
NLI_INT_TO_STR: Dict[int, str] = {0: "entailment", 1: "neutral", 2: "contradiction"}


def normalize_text(value) -> str:
    """Convert scalar/list/dict dataset fields into stable plain text."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, list):
        return "\n\n".join(filter(None, (normalize_text(v) for v in value)))
    if isinstance(value, dict):
        return "\n\n".join(filter(None, (normalize_text(v) for v in value.values())))
    return " ".join(str(value).split())


def load_config(config_path: str) -> Dict:
    return load_project_config(config_path)


def validate_record(row: pd.Series, idx: int) -> Optional[str]:
    for col in REQUIRED_COLUMNS:
        val = row.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return f"Row {idx}: missing '{col}'"
        if isinstance(val, str) and not val.strip():
            return f"Row {idx}: empty string in '{col}'"
    try:
        numeric_label = float(row["hallu_label"])
        if not numeric_label.is_integer():
            return f"Row {idx}: hallu_label='{row['hallu_label']}' is fractional"
        lbl = int(numeric_label)
    except (ValueError, TypeError):
        return f"Row {idx}: non-integer hallu_label='{row['hallu_label']}'"
    if lbl not in (0, 1):
        return f"Row {idx}: hallu_label={lbl} not in {{0,1}}"
    if (
        "nli_label" in row
        and row.get("nli_label") is not None
        and not (isinstance(row["nli_label"], float) and np.isnan(row["nli_label"]))
    ):
        try:
            numeric_nli = float(row["nli_label"])
            if not numeric_nli.is_integer():
                return f"Row {idx}: nli_label='{row['nli_label']}' is fractional"
            nli = int(numeric_nli)
        except (ValueError, TypeError):
            return f"Row {idx}: non-integer nli_label='{row['nli_label']}'"
        if nli not in (0, 1, 2):
            return f"Row {idx}: nli_label={nli} not in {{0,1,2}}"
    return None


def map_record(row: pd.Series) -> Dict:
    raw_id = row.get("id", -1)
    if raw_id is None or (isinstance(raw_id, float) and np.isnan(raw_id)):
        raw_id = -1
    if raw_id == -1 or not str(raw_id).strip():
        identity = "\n".join(
            str(row[name]).strip() for name in ("question", "context", "answer", "hallu_label")
        )
        raw_id = "auto-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    record = {
        # IDs are identifiers, not quantities.  Preserve string/UUID IDs instead
        # of crashing by forcing every dataset's ID through int(...).
        "sample_id": raw_id.item() if hasattr(raw_id, "item") else raw_id,
        "question": str(row["question"]).strip(),
        "context": str(row["context"]).strip(),
        "answer": str(row["answer"]).strip(),
        "reason": str(row.get("reason", "")).strip(),
        "hall_label": int(float(row["hallu_label"])),
    }
    nli_raw = row.get("nli_label")
    if nli_raw is not None and not (isinstance(nli_raw, float) and np.isnan(nli_raw)):
        nli_int = int(float(nli_raw))
        record["nli_label"] = nli_int
        record["nli_label_str"] = NLI_INT_TO_STR.get(nli_int, "neutral")
    for key in OPTIONAL_COLUMNS - {"reason", "nli_label"}:
        value = row.get(key)
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            record[key] = str(value).strip()
    return record


def canonical_group_key(record: Dict) -> str:
    """Stable claim-family key used to keep factual/hallucinated pairs together.

    MedHallu-style data commonly contains two answers for one question/context.
    Splitting rows independently leaks the paired example into another split and
    produces optimistic metrics.  IDs cannot be trusted (and eval data may not
    have them), so group on normalized question + context content.
    """

    def norm(value: str) -> str:
        return " ".join(str(value).casefold().split())

    explicit_group = str(record.get("group_id", "")).strip()
    if explicit_group:
        payload = "group_id\n" + explicit_group.casefold()
    else:
        payload = norm(record["question"]) + "\n" + norm(record["context"])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_group_disjoint(*named_splits: Tuple[str, List[Dict]]) -> None:
    keys = {
        name: {canonical_group_key(record) for record in records} for name, records in named_splits
    }
    names = list(keys)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = keys[left] & keys[right]
            if overlap:
                raise RuntimeError(
                    f"Data leakage: {len(overlap)} question/context group(s) "
                    f"occur in both {left} and {right}."
                )


def cross_split_overlap_report(
    *named_splits: Tuple[str, List[Dict]],
) -> Dict[str, Dict[str, object]]:
    """Return pairwise question/context overlap counts and example hashes."""
    keys = {
        name: {canonical_group_key(record) for record in records} for name, records in named_splits
    }
    report: Dict[str, Dict[str, object]] = {}
    names = list(keys)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = sorted(keys[left] & keys[right])
            report[f"{left}__{right}"] = {
                "count": len(overlap),
                "example_group_hashes": overlap[:20],
            }
    return report


def remove_cross_split_overlap(
    splits: Dict[str, List[Dict]],
) -> Tuple[Dict[str, List[Dict]], List[Dict]]:
    """Make fixed splits disjoint using eval > val > train priority.

    No row is moved or resampled. Rows from a lower-priority split are removed
    when their normalized question+context group already occurs in a protected
    higher-priority split. Removed rows are returned for an audit JSONL.
    """
    cleaned = {name: list(rows) for name, rows in splits.items()}
    removed: List[Dict] = []

    eval_keys = {canonical_group_key(r) for r in cleaned["eval"]}
    kept_val = []
    for record in cleaned["val"]:
        key = canonical_group_key(record)
        if key in eval_keys:
            removed.append(
                {**record, "removed_from": "val", "overlaps_with": "eval", "group_hash": key}
            )
        else:
            kept_val.append(record)
    cleaned["val"] = kept_val

    protected_keys = eval_keys | {canonical_group_key(r) for r in cleaned["val"]}
    kept_train = []
    for record in cleaned["train"]:
        key = canonical_group_key(record)
        if key in protected_keys:
            target = "eval" if key in eval_keys else "val"
            removed.append(
                {**record, "removed_from": "train", "overlaps_with": target, "group_hash": key}
            )
        else:
            kept_train.append(record)
    cleaned["train"] = kept_train

    for name in cleaned:
        cleaned[name] = assign_row_idx(cleaned[name])
    assert_group_disjoint(*cleaned.items())
    return cleaned, removed


def load_and_validate(csv_path: str) -> List[Dict]:
    separator = "\t" if Path(csv_path).suffix.lower() == ".tsv" else ","
    logger.info("Reading delimited file: %s", csv_path)
    df = pd.read_csv(csv_path, dtype=str, sep=separator)
    return _validate_dataframe(df, csv_path)


def load_and_validate_json(json_path: str) -> List[Dict]:
    logger.info("Reading JSON: %s", json_path)
    with open(json_path, "r", encoding="utf-8") as fh:
        raw_records = json.load(fh)
    if not isinstance(raw_records, list):
        raise ValueError(f"{json_path} must contain a top-level JSON array of records.")
    df = pd.DataFrame(raw_records)
    return _validate_dataframe(df, json_path)


def load_and_validate_jsonl(jsonl_path: str) -> List[Dict]:
    """Load newline-delimited JSON without changing file membership/order."""
    logger.info("Reading JSONL: %s", jsonl_path)
    rows = []
    with open(jsonl_path, "r", encoding="utf-8-sig") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{jsonl_path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{jsonl_path}:{line_no}: each line must be a JSON object")
            rows.append(item)
    return _validate_dataframe(pd.DataFrame(rows), jsonl_path)


def load_input(path: str) -> List[Dict]:
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        return load_and_validate_jsonl(path)
    if suffix == ".json":
        return load_and_validate_json(path)
    if suffix in (".csv", ".tsv"):
        return load_and_validate(path)
    raise ValueError(f"Unsupported input extension '{suffix}' for {path}")


def _validate_dataframe(df: pd.DataFrame, source_path: str) -> List[Dict]:
    if df.empty:
        raise ValueError(f"{source_path} contains no records")
    df.columns = [c.strip().lower() for c in df.columns]
    # Accept common MedHallu/PubMedQA field names directly. Canonical fields
    # always win when both canonical and alias columns are present.
    rename = {}
    for canonical, aliases in FIELD_ALIASES.items():
        if canonical not in df.columns:
            alias = next((name for name in aliases if name in df.columns), None)
            if alias is not None:
                rename[alias] = canonical
    if rename:
        logger.info("Normalizing input fields: %s", rename)
        df = df.rename(columns=rename)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{source_path} missing required fields: {missing}")

    if "reason" not in df.columns:
        logger.info(
            "No 'reason' field in %s — the JSONL 'reason' field will be "
            "written as an empty string for every record.",
            source_path,
        )

    for column in ("question", "answer", "context"):
        df[column] = df[column].map(normalize_text)

    identity_columns = ["question", "answer", "context"]
    conflicts = df.groupby(identity_columns, dropna=False)["hallu_label"].nunique(dropna=False)
    if int((conflicts > 1).sum()):
        raise ValueError(
            f"{source_path} contains {int((conflicts > 1).sum())} identical "
            "question/answer/context group(s) with conflicting hallu_label values. "
            "Resolve annotation conflicts before preprocessing."
        )
    duplicate_count = int(
        df.duplicated(
            subset=identity_columns + ["hallu_label"],
            keep=False,
        ).sum()
    )
    if duplicate_count:
        raise ValueError(
            f"{source_path} contains {duplicate_count} duplicate rows. Fixed-split "
            "preprocessing never removes rows silently; deduplicate the source file."
        )
    records, errors = [], []
    for idx, row in df.iterrows():
        err = validate_record(row, idx)
        if err:
            errors.append(err)
        else:
            records.append(map_record(row))
    if errors:
        preview = "\n".join(f"  {e}" for e in errors[:5])
        raise ValueError(
            f"{source_path} has {len(errors)} invalid record(s). Fix the input "
            f"instead of silently dropping data. First errors:\n{preview}"
        )
    logger.info("Valid records: %d / %d", len(records), len(df))
    return records


def assign_row_idx(records: List[Dict]) -> List[Dict]:
    for i, rec in enumerate(records):
        rec["row_idx"] = i
    return records


def write_jsonl(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records → %s", len(records), path)


def log_split_stats(name: str, records: List[Dict]) -> None:
    n = len(records)
    h = Counter(r["hall_label"] for r in records)
    logger.info(
        "%s n=%d | hall: 0(not_hall)=%d(%.1f%%) 1(hall)=%d(%.1f%%)",
        name.ljust(6),
        n,
        h[0],
        h[0] / n * 100,
        h[1],
        h[1] / n * 100,
    )


def run_fixed_preprocessing(
    config_path: str,
    train_input: str,
    val_input: str,
    eval_input: str,
    overlap_policy: Optional[str] = None,
) -> None:
    """Normalize three user-defined splits; never sample or re-split records."""
    cfg = load_config(config_path)
    processed_dir = Path(cfg["data"]["processed_dir"])
    splits = {
        "train": assign_row_idx(load_input(train_input)),
        "val": assign_row_idx(load_input(val_input)),
        "eval": assign_row_idx(load_input(eval_input)),
    }
    for name, rows in splits.items():
        if not rows:
            raise ValueError(f"{name} input contains no valid records")
        log_split_stats(name.upper(), rows)

    # Fixed files are authoritative: never silently move, delete, or resplit.
    policy = (
        overlap_policy
        or cfg["data"].get("overlap_policy")
        or cfg["data"].get("fixed_split_overlap_policy", "warn")
    ).lower()
    if policy not in {"remove_lower_priority", "warn", "error", "ignore"}:
        raise ValueError(
            "fixed_split_overlap_policy must be remove_lower_priority, warn, error, or ignore"
        )
    overlap_report = cross_split_overlap_report(*splits.items())
    overlap_total = sum(int(item["count"]) for item in overlap_report.values())
    removed_overlap_rows: List[Dict] = []
    original_counts = {name: len(rows) for name, rows in splits.items()}
    if overlap_total and policy == "remove_lower_priority":
        splits, removed_overlap_rows = remove_cross_split_overlap(splits)
        logger.warning(
            "Removed %d lower-priority row(s) to eliminate cross-split "
            "question/context leakage (priority: eval > val > train).",
            len(removed_overlap_rows),
        )
        for name, rows in splits.items():
            if not rows:
                raise RuntimeError(
                    f"Cross-split cleanup left {name} empty. Rebuild the source "
                    "files with disjoint question/context groups."
                )
    if overlap_total and policy == "error":
        assert_group_disjoint(*splits.items())
    if overlap_total and policy == "warn":
        summary = ", ".join(
            f"{pair.replace('__', '/')}={item['count']}"
            for pair, item in overlap_report.items()
            if item["count"]
        )
        logger.warning(
            "Cross-split question/context overlap detected (%s). Preserving "
            "the supplied split membership and continuing because overlap "
            "policy is 'warn'. Validation/final metrics may be optimistic.",
            summary,
        )
    output_names = {
        "train": "train.jsonl",
        "val": "validation.jsonl",
        "eval": "test.jsonl",
    }
    for name, rows in splits.items():
        write_jsonl(rows, processed_dir / output_names[name])
    if removed_overlap_rows:
        write_jsonl(
            removed_overlap_rows,
            processed_dir / "removed_cross_split_overlap.jsonl",
        )
    else:
        # Do not leave a stale audit report from an earlier preprocessing run.
        stale_audit = processed_dir / "removed_cross_split_overlap.jsonl"
        if stale_audit.exists():
            stale_audit.unlink()

    meta = {
        "protocol": "fixed_user_provided_splits_no_resampling",
        "sources": {"train": train_input, "val": val_input, "eval": eval_input},
        "counts": {name: len(rows) for name, rows in splits.items()},
        "original_counts_before_cross_split_cleanup": original_counts,
        "removed_cross_split_rows": len(removed_overlap_rows),
        "label_counts": {
            name: {str(k): int(v) for k, v in Counter(r["hall_label"] for r in rows).items()}
            for name, rows in splits.items()
        },
        "cross_split_overlap_policy": policy,
        "cross_split_overlap": overlap_report,
        "usage": {
            "train": "parameter optimization only",
            "val": "early stopping, checkpoint selection, temperature and threshold only",
            "eval": "one-time final evaluation only",
        },
    }
    processed_dir.mkdir(parents=True, exist_ok=True)
    with open(processed_dir / "preprocessing_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("Fixed-split preprocessing complete; no data splitting was performed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--train_input", type=str)
    parser.add_argument("--val_input", type=str)
    parser.add_argument("--eval_input", type=str)
    parser.add_argument(
        "--overlap_policy",
        choices=("remove_lower_priority", "warn", "error", "ignore"),
        default=None,
        help="How fixed-split question/context overlap is handled. Default is "
        "the config value (remove_lower_priority in the supplied config).",
    )
    args = parser.parse_args()
    fixed = (args.train_input, args.val_input, args.eval_input)
    if not all(fixed):
        parser.error("Fixed-split protocol requires --train_input, --val_input and --eval_input")
    run_fixed_preprocessing(args.config, *fixed, overlap_policy=args.overlap_policy)
