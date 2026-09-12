# Resuming interrupted runs

Every long-running command in ClinHallu resumes automatically. If a run stops
for any reason — a Colab disconnect, a dropped network, a closed laptop, a power
cut, `Ctrl-C`, an out-of-memory kill — **re-run the identical command** and it
continues from where it stopped instead of starting over.

```bash
clinhallu ablate --variants B C D E F --seed 13   # stops during variant D
clinhallu ablate --variants B C D E F --seed 13   # skips B and C, continues D mid-epoch
```

There is no separate "resume" command and no flag to remember. Resuming is the
default; `--no-resume` is how you deliberately start over.

---

## 1. What gets saved, and how often

| Level | Saved | Worst case if the machine dies |
|---|---|---|
| Optimizer step | every `save_every_steps` steps or `save_every_minutes` minutes | that many steps |
| Epoch | after the training pass, and again after scoring | nothing |
| Experiment stage | after prepare / train / evaluate | the stage in progress, which itself resumes |
| Ablation variant | after each of B–F | the variant in progress, which itself resumes |
| Baseline | after each baseline | the baseline in progress |
| Pipeline stage | after each of the five stages | the stage in progress |

Defaults, in `configs/base.yaml`:

```yaml
checkpoints:
  save_every_steps: 200
  save_every_minutes: 10
```

Whichever limit is reached first triggers a save, so a slow epoch is still
covered by the time bound. Lower them on flaky connections; raise them if
checkpoint writing is eating into throughput on a slow disk (Google Drive).

---

## 2. What a checkpoint contains

Resuming is *equivalent*, not approximate. A resumed run produces bit-identical
weights to one that was never interrupted, because the checkpoint carries
everything that affects the trajectory:

- model, optimizer, LR-scheduler and AMP `GradScaler` state
- `global_step`, epoch number, best metric so far, early-stopping patience
- position inside the epoch (`batches_done`, `samples_done`)
- partially accumulated labels, probabilities, logits and loss for that epoch
- RNG state for Python `random`, NumPy, PyTorch CPU and PyTorch CUDA
- the full per-epoch metrics history

The equivalence is enforced by a test
(`tests/integration/test_training_resume.py::test_resumed_epoch_matches_an_uninterrupted_epoch`)
that trains an epoch twice — once straight through, once killed half way — and
asserts the weights match exactly.

### Why the order of examples is deterministic

`DataLoader(shuffle=True)` draws from a generator that advances every epoch, so
re-seeding at process start replays epoch 1's order no matter which epoch you
resume into. That silently revisits some examples and skips others.

`ResumableRandomSampler` instead derives the permutation from a hash of
`(seed, epoch)`. Epoch *k* is always the same order, so "continue 3,200 examples
in" is exact: the examples before the crash and the examples after it partition
the epoch with no overlap and no gaps.

---

## 3. Checkpoints survive a power cut

Writes are durable rather than merely quick:

1. write to `latest_checkpoint.pt.tmp`
2. `fsync` the file so the bytes reach the disk
3. rename over the target (atomic on POSIX)
4. `fsync` the directory

A crash therefore leaves either the previous checkpoint or the new one, never a
half-written file.

Before each write the current checkpoint is rotated to
`latest_checkpoint.pt.bak`. If the newest file turns out to be unreadable —
a filesystem that lies about `fsync`, a truncated Drive sync — the run falls
back one interval instead of restarting from epoch 1, and logs a warning. If
*both* are unreadable the run raises rather than silently discarding your work:

```
RuntimeError: Every resume checkpoint is unreadable, so continuing would
silently discard prior work. Inspect or delete these files and re-run:
```

---

## 4. Stopping cleanly on purpose

`Ctrl-C` (or `SIGTERM`, or `SIGHUP` when a terminal closes) no longer kills the
process mid-batch. The signal is recorded, the current optimizer step finishes,
a checkpoint is written, and the process exits with code **130**:

```
SIGINT received. Finishing the current step, saving a checkpoint, then
stopping. Re-run the same command to continue.
```

Press `Ctrl-C` a second time to force an immediate exit — you lose only the
partial step since the last save.

Exit code 130 means "stopped cleanly, progress saved" and is propagated up
through the ablation runner and pipeline, so those distinguish a clean stop from
a genuine failure in their ledgers.

Nothing can protect against `SIGKILL` or a yanked power cable. Those are covered
by the checkpoint cadence in section 1.

---

## 5. Multi-stage jobs: the ledger

A training checkpoint makes one run resumable. A **ledger** makes a *sequence*
of runs resumable. Each records per-unit status the instant it changes:

| Job | Ledger |
|---|---|
| One experiment's stages | `artifacts/<kind>/<name>/seed_<n>/stage_ledger.json` |
| Ablation variants B–F | `artifacts/ablations/ablation_ledger.json` |
| Baselines | `artifacts/baselines/baseline_ledger.json` |
| Pipeline stages | `artifacts/pipeline/<condition>_seed<n>.json` |

Each entry holds a status (`pending` / `running` / `completed` / `failed` /
`interrupted`), an attempt count, timestamps, and a **fingerprint** of the
inputs.

The fingerprint is what makes skipping safe. If you edit a learning rate, the
fingerprint changes, the old `completed` entry no longer matches, and the unit
runs again rather than reusing a stale result:

```
Unit 'B' completed previously, but its configuration changed
(a1b2c3d4 -> e5f6a7b8); it will run again.
```

Entries written before fingerprinting existed are trusted rather than discarded,
so upgrading does not throw away finished work.

---

## 6. Checking progress before you commit a GPU session

```bash
clinhallu status
```

```
Project: /content/drive/MyDrive/clinhallu

artifacts/adaptation/c5/seed_13
  stages: prepare=completed, train=running, evaluate=pending
  latest checkpoint: epoch 3 (mid-epoch at batch 812), step 1420, best=0.7314
  evaluated: no

ablations: {"completed": 2, "interrupted": 1}
```

`clinhallu status --json` emits the same thing as JSON. For pipeline runs
specifically:

```bash
clinhallu pipeline --condition c5 --seed 13 --status
```

---

## 7. Deliberately re-running things

| Goal | Command |
|---|---|
| Retrain one experiment from scratch | `clinhallu condition c5 --seed 13 --no-resume` |
| Re-run all ablations from scratch | `clinhallu ablate --variants B C D E F --no-resume` |
| Re-run only the evaluation stage | `clinhallu condition c5 --force-stage evaluate` |
| Re-run one pipeline stage | `clinhallu pipeline --force-stage baselines` |
| Re-run one baseline | `clinhallu baseline run --only selfcheckgpt --no-resume` |
| Keep going past a failing ablation | `clinhallu ablate --variants B C D E F --continue-on-error` |

`--no-resume` clears the relevant ledger entries and ignores existing training
checkpoints. It does not delete run directories, so previous artifacts remain on
disk until overwritten.

---

## 8. Colab specifics

Run from a Drive-mounted directory so checkpoints outlive the VM:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/clinhallu
```

Drive syncs asynchronously and is slow for large writes. If checkpoint saves feel
like they are throttling training, raise the interval:

```yaml
checkpoints:
  save_every_steps: 500
  save_every_minutes: 20
```

When a Colab session is reclaimed, reconnect, re-mount Drive, re-install, and
re-run the identical command. `clinhallu status` first will show what survived.

A twelve-hour Colab limit is not a problem: run the same command across as many
sessions as it takes.

---

## 9. Things resume does *not* paper over

- **A changed config mid-run.** The fingerprint catches this for ledger units,
  and `find_latest_checkpoint` refuses a checkpoint whose state keys do not
  match the current architecture. Use a new experiment name instead.
- **Changed data.** The runner already compares the raw-split SHA-256 against
  `data/manifests/` and refuses to continue if it moved.
- **A genuine bug.** A failure recorded as `failed` is retried on the next run;
  if it fails the same way, fix the cause rather than re-running.
- **Frozen-test integrity.** Resume never touches the evaluation split. The
  smoke-test path still refuses to evaluate the frozen test set.

---

## 10. Where the behaviour is tested

| File | Covers |
|---|---|
| `tests/unit/test_resumable_sampler.py` | deterministic order, exact skip, no overlap or gaps |
| `tests/unit/test_ledger.py` | durability, fingerprint invalidation, retry of failures |
| `tests/regression/test_checkpoint_resume.py` | phases, backup fallback, RNG round-trip, `weights_only` loading |
| `tests/integration/test_training_resume.py` | interrupted epoch produces bit-identical weights |
| `tests/integration/test_pipeline_resume.py` | ablations and pipeline skip completed units |
| `tests/integration/test_crash_recovery.py` | real tiny-model training interrupted and resumed end to end |

```bash
pytest tests/unit/test_resumable_sampler.py tests/unit/test_ledger.py \
       tests/regression/test_checkpoint_resume.py \
       tests/integration/test_training_resume.py \
       tests/integration/test_pipeline_resume.py \
       tests/integration/test_crash_recovery.py
```

---

## 11. See it for yourself

`scripts/demo_resume.py` builds a self-contained toy project, starts a real
training subprocess, sends it `SIGINT` once a mid-epoch checkpoint lands, then
re-runs the identical command and checks that it continued:

```bash
python scripts/demo_resume.py
```

```
Attempt 1: starting training, interrupting once a mid-epoch save lands...
  exit code: 130 (130 == stopped cleanly, progress saved)
  | Checkpoint written mid-epoch 1 at batch 55/120 (step 55).
  | Stopped during epoch 1 at batch 55/120. Re-run the same command to continue.
  checkpoint: {'epoch': 1, 'phase': 'in_epoch', 'global_step': 55, 'batches_done': 55}

Attempt 2: re-running the identical command...
  exit code: 0
  | Resuming inside epoch 1 after 55 batches (110 examples, step=55)

  epochs recorded: 3 (expected 3)
  best checkpoint: True
  calibration:     True

PASS: interrupted training resumed and completed.
```

No network and no GPU required; it takes about a minute.
