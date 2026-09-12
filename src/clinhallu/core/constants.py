"""Project-wide names that define the publication protocol."""

LABEL_GROUNDED = 0
LABEL_HALLUCINATED = 1
DEFAULT_SEED = 13
RAW_SPLIT_FILENAMES = {
    "train": "train.jsonl",
    "validation": "val.jsonl",
    "test": "eval_data.jsonl",
}
PROCESSED_SPLIT_FILENAMES = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "test": "test.jsonl",
}
