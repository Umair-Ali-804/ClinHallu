# Data placement

All five conditions share one held-out validation split and one frozen
evaluation split. Only the training file changes between conditions.

```text
data/raw/c1/train.jsonl      10,000 rows
data/raw/c2/train.jsonl      11,000 rows
data/raw/c3/train.jsonl      12,000 rows
data/raw/c4/train.jsonl      13,000 rows
data/raw/train.jsonl         14,000 rows   (C5)
data/raw/val.jsonl            4,000 rows   (shared, MedHallu-derived target split)
data/raw/eval_data.jsonl      1,000 rows   (shared, frozen MedHallu-derived expert test split)
```

## Condition composition

The conditions are exact cumulative supersets. C1 is the 10,000-example
PubMedQA-derived source pool. Each later condition keeps every earlier record
and adds exactly 1,000 MedHallu-derived target records.

| Condition | Source rows | Target rows | Total |
|---|---|---|---|
| C1 | 10,000 | 0 | 10,000 |
| C2 | 10,000 | 1,000 | 11,000 |
| C3 | 10,000 | 2,000 | 12,000 |
| C4 | 10,000 | 3,000 | 13,000 |
| C5 | 10,000 | 4,000 | 14,000 |

Every split is class-balanced (positive rate 0.500). The expert test set has
500 supported and 500 hallucinated examples.

C5 reads `data/raw/train.jsonl` rather than a `c5/` subdirectory because the
ablation configs inherit their training path from `configs/base.yaml`.

## Fields

Required fields are `question`, `context`, `answer`, and `hallu_label`
(0 = supported, 1 = hallucinated). Use a unique `id` whenever possible;
external judge baselines require it. The model pipeline generates a stable
content-derived ID if it is absent. The code accepts documented legacy aliases
but writes one canonical processed schema.

## Rules

- Never put evaluation examples into training or validation. Question/context
  groups must not overlap across train, validation, and evaluation.
- Validation and evaluation must come from the target (MedHallu-derived)
  distribution in every condition, including C1.
- Set each condition's audited row count in `configs/conditions/cN.yaml` under
  `data.expected_rows.train` before the first run. Do not pad a split with
  duplicates to reach a round number.
- Do not manually edit `data/processed/`, `data/cache/`, or `data/manifests/`.
