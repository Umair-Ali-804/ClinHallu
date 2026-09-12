# Data placement

All five conditions share one held-out validation split and one frozen
evaluation split. Only the training file changes between conditions.

```text
data/raw/c1/train.jsonl      10,000 rows
data/raw/c2/train.jsonl      11,000 rows
data/raw/c3/train.jsonl      12,000 rows
data/raw/c4/train.jsonl      13,000 rows
data/raw/train.jsonl         14,000 rows   (C5)
data/raw/val.jsonl            4,000 rows   (shared, MedHallu artificial split)
data/raw/eval_data.jsonl      1,000 rows   (shared, frozen MedHallu expert split)
```

C5 reads `data/raw/train.jsonl` rather than a `c5/` subdirectory because the
ablation configs inherit their training path from `configs/base.yaml`.

Required fields are `question`, `context`, `answer`, and `hallu_label`.
Use a unique `id` whenever possible; external judge baselines require it. The
model pipeline generates a stable content-derived ID if it is absent. The code
accepts documented legacy aliases but writes one canonical processed schema.
Never put evaluation examples into training or validation.

Set each condition's audited row count in `configs/conditions/cN.yaml` under
`data.expected_rows.train` before the first run. Do not pad a split with
duplicates to reach a round number.

Do not manually edit `data/processed/`, `data/cache/`, or `data/manifests/`.
