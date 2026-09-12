# Fixed-split data protocol

The project consumes three user-provided JSONL files and never creates a new
random split. Label `0` means grounded and label `1` means hallucinated.

| File | Permitted use |
|---|---|
| `data/raw/train.jsonl` | Parameter optimization only |
| `data/raw/val.jsonl` | Early stopping, checkpoint selection, calibration and threshold |
| `data/raw/eval_data.jsonl` | Frozen final evaluation only |

The validator normalizes documented field aliases, rejects invalid labels,
checks both classes, enforces configured row counts, hashes each file, and
rejects overlapping question/context families across splits.

Run before an expensive experiment:

```bash
clinhallu validate --config configs/conditions/c5.yaml
```
