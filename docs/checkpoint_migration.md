# Checkpoint migration

The refactor preserves GAER++ and legacy model attribute names, so state-dict
keys from the supplied project remain valid.

Copying is safer than moving. The migration script copies old files into a new
run directory and leaves the source untouched:

```bash
python scripts/migrate_old_checkpoints.py \
  --old-checkpoint-dir checkpoints/distributed_study/pubmed10k_medhallu4k/seed_13 \
  --new-run-dir artifacts/adaptation/c5/seed_13
```

The new evaluator expects:

```text
checkpoints/best_model.pt
calibration/temperature.npy
calibration/threshold.npy
```

Do not rename parameters inside a checkpoint. Always verify migrated weights
with `scripts/verify_results.py` before deleting or archiving the old project.

## Checkpoints written before the resume rewrite

The checkpoint format gained a `format_version` field (currently 2) along with
step-level resume state: `phase`, `batches_done`, `samples_done`,
`epoch_progress`, `history` and `rng_state`.

Older checkpoints are read without conversion. A payload with no `phase` field
is treated as a completed epoch, so a resumed run starts at the following epoch
— exactly the behaviour it had before. You lose only the finer granularity,
which the next checkpoint restores:

```bash
clinhallu condition c5 --seed 13    # picks up at the next epoch, then saves v2
```

No migration step is required and no existing run needs to be retrained.

Two files now exist per slot: `latest_checkpoint.pt` and
`latest_checkpoint.pt.bak`, the previous save. The backup is only read when the
newest file is unreadable. Keep both; deleting the backup removes the fallback
that protects against a truncated or interrupted write.
