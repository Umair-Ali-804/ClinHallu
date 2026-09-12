# ClinHallu submission guide

What to submit alongside the paper, where each file belongs, the exact commands
that produce it, and how to publish the repository on GitHub.

This guide assumes the five-condition design: C1–C5 differ only in training set
size (10k / 11k / 12k / 13k / 14k) and share one 4,000-row validation split and
one frozen 1,000-row evaluation split.

---

## 1. Three things a reviewer asks for

| Deliverable | Goes where | Size |
|---|---|---|
| **Code** | GitHub repository | ~1.3 MB |
| **Data** | Zenodo / OSF archive, or a licence-gated request link | ~100–200 MB |
| **Results** | Small files in the repo; large checkpoints in a Zenodo archive | see §5 |

Only code is small enough to live entirely in Git. Split the other two out and
link to them from the README. Do not try to put trained checkpoints in Git; see
§5.3 for why.

---

## 2. Where every file goes

### 2.1 Input data (you provide)

```text
data/raw/c1/train.jsonl        10,000 rows
data/raw/c2/train.jsonl        11,000 rows
data/raw/c3/train.jsonl        12,000 rows
data/raw/c4/train.jsonl        13,000 rows
data/raw/train.jsonl           14,000 rows      C5
data/raw/val.jsonl              4,000 rows      shared by all five
data/raw/eval_data.jsonl        1,000 rows      shared, frozen
```

C5 reads `data/raw/train.jsonl` and **not** a `c5/` subfolder. This is
deliberate: the five ablation configs inherit their training path from
`configs/base.yaml`. If you move C5's file without also editing
`configs/base.yaml`, `clinhallu ablate` stops with *"Data registered as 'c5'
changed"*.

Each line is one JSON object:

```json
{"id":"example_001","question":"...","context":"...","answer":"...","hallu_label":0}
```

`hallu_label` is 0 for supported and 1 for hallucinated. IDs must be unique
inside a split, and question/context groups must not cross splits.

If your audited row counts are not exactly 10k/11k/12k/13k/14k, set the real
number in each `configs/conditions/cN.yaml` under `data.expected_rows.train`
before the first run. Never pad a split with duplicates to hit a round number.

### 2.2 Prompts (already in the repo)

```text
prompts/shared_five_system.txt      answer-generation system prompt
prompts/shared_five_user.txt        answer-generation user prompt
prompts/judge_system.txt            LLM-judge system prompt
prompts/judge_user.txt              LLM-judge user prompt
```

These are the exact prompts used for the paper. Ship them unchanged — a
reviewer checking prompt sensitivity will look here first.

### 2.3 Generated artefacts (the code writes these)

```text
artifacts/adaptation/<cN>/seed_13/            one per condition
artifacts/ablations/<variant>/seed_13/        B, C, D, E, F
artifacts/baselines/<instance>/               13 configured baselines
artifacts/reports/tables/                     CSV for the paper tables
artifacts/reports/figures/                    PNG + PDF
artifacts/reports/summaries/                  Markdown tables
data/generated/shared_five/                   the five sampled answers
```

Each run directory contains:

```text
checkpoints/best_model.pt              full resumable checkpoint  (large)
checkpoints/latest_checkpoint.pt       resume point               (large)
checkpoints/model_only.safetensors     weights only               (large)
calibration/temperature.npy            validation-fitted temperature
calibration/threshold.npy              validation-selected threshold
calibration/calibration.json
metrics/evaluation_metrics.json        frozen-test numbers
metrics/validation_metrics.json
metrics/training_history.csv           per-epoch loss and F1
predictions/evaluation_predictions.jsonl
resolved_config.yaml                   every setting, fully resolved
run_manifest.json                      dataset hashes, protocol version
environment.json                       library versions, GPU
logs/
```

`resolved_config.yaml`, `run_manifest.json` and `environment.json` are what make
the run auditable. Include them in the results archive even though they are tiny.

---

## 3. The exact command sequence

Run from the project root, with `OPENROUTER_API_KEY` exported.

```bash
# 0. environment
python -m pip install -e '.[dev]'
python scripts/check_environment.py

# 1. confirm every split before spending GPU time
clinhallu validate --config configs/conditions/c1.yaml
clinhallu validate --config configs/conditions/c2.yaml
clinhallu validate --config configs/conditions/c3.yaml
clinhallu validate --config configs/conditions/c4.yaml
clinhallu validate --config configs/conditions/c5.yaml

# 2. the five conditions
clinhallu condition c1 --seed 13
clinhallu condition c2 --seed 13
clinhallu condition c3 --seed 13
clinhallu condition c4 --seed 13
clinhallu condition c5 --seed 13

# 3. the five ablations (C5 data)
clinhallu ablate --variants B C D E F --seed 13

# 4. generate the five shared answers once — check the cost first
clinhallu generate --config configs/generation/shared_five.yaml --dry-run
clinhallu generate --config configs/generation/shared_five.yaml

# 5. free / local baselines
clinhallu baseline run --only tfidf_lr
clinhallu baseline run --only semantic_similarity
clinhallu baseline run --only zero_shot_nli
clinhallu baseline run --only selfcheckgpt
clinhallu baseline run --only self_consistency

# 6. paid judges, one at a time
clinhallu baseline run --only openrouter_gpt_5_mini
clinhallu baseline run --only openrouter_gemini_2_5_flash
clinhallu baseline run --only openrouter_deepseek_chat
clinhallu baseline run --only openrouter_llama_3_3_70b_instruct
clinhallu baseline run --only openrouter_qwen3_32b
clinhallu baseline run --only openrouter_llama_4_maverick
clinhallu baseline run --only openrouter_qwen3_30b_a3b
clinhallu baseline run --only openrouter_gemini_2_5_flash_lite

# 7. build every table and figure
clinhallu report build

# 8. audit what you are about to publish
python scripts/verify_results.py artifacts/adaptation/*/seed_13
clinhallu status
```

Every one of these resumes. If a run stops for any reason, re-run the identical
command and it continues from its checkpoint.

**Budget check before step 4.** `--dry-run` prints
`unique_question_context_pairs` and `new_requests_needed`. For a 1,000-row
evaluation split expect 1,000 pairs and 5,000 completions, because each pair is
sampled five times with seeds 13–17.

For additional seeds, repeat steps 2–3 with `--seed 42` and `--seed 7`. A
three-seed mean with a standard deviation is much harder for a reviewer to
argue with than a single run.

---

## 4. Mapping results to paper tables

| Paper element | File | Command |
|---|---|---|
| Main results, C1–C5 | `artifacts/reports/tables/adaptation_results.csv` | `report build` |
| Main results, formatted | `artifacts/reports/summaries/main_results.md` | `report build` |
| Ablation table B–F | `artifacts/reports/tables/ablation_results.csv` | `report build` |
| Baseline comparison | `artifacts/reports/tables/baseline_results.csv` | `report build` |
| ROC curve | `artifacts/reports/figures/roc_curve.pdf` | `report build` |
| PR curve | `artifacts/reports/figures/precision_recall_curve.pdf` | `report build` |
| Reliability diagram | `artifacts/reports/figures/reliability_diagram.pdf` | `report build` |
| Confusion matrix | `artifacts/reports/figures/confusion_matrix.pdf` | `report build` |
| Learning curve | `artifacts/reports/figures/learning_curve.pdf` | `report build` |
| Per-condition raw numbers | `artifacts/adaptation/<cN>/seed_13/metrics/evaluation_metrics.json` | `condition cN` |

`adaptation_results.csv` already contains one row per condition with accuracy,
precision, recall, F1, MCC, AUROC, AUPRC, Brier and calibrated ECE. That single
file is your main results table.

Use the PDF figures in the manuscript, not the PNGs — they stay sharp at any
zoom level. Metrics are fractions, not percentages; multiply by 100 yourself if
the venue wants percentages, and say so in the caption.

Report the grouped bootstrap F1 interval (`eval_f1_ci_low` / `eval_f1_ci_high`)
next to the point estimate. Reviewers increasingly expect it.

---

## 5. What to publish, and where

### 5.1 GitHub repository — code only

Everything already tracked in Git: `src/`, `tests/`, `configs/`, `prompts/`,
`scripts/`, `notebooks/`, `docs/`, `README.md`, `pyproject.toml`,
`requirements/`, `LICENSE`, `CITATION.cff`. About 1.3 MB.

### 5.2 Small results — safe to commit

These are worth putting in the repo so a reader sees the numbers without
downloading anything:

```text
artifacts/reports/tables/       a few KB of CSV
artifacts/reports/summaries/    Markdown tables
artifacts/reports/figures/      PDF + PNG, a few MB
```

`.gitignore` currently excludes all of `artifacts/`. To commit just the reports:

```bash
git add -f artifacts/reports/tables artifacts/reports/summaries artifacts/reports/figures
```

Also worth force-adding, one per run — they are tiny and make the work auditable:

```bash
git add -f artifacts/adaptation/*/seed_13/metrics/evaluation_metrics.json
git add -f artifacts/adaptation/*/seed_13/resolved_config.yaml
git add -f artifacts/adaptation/*/seed_13/run_manifest.json
git add -f artifacts/ablations/*/seed_13/metrics/evaluation_metrics.json
```

### 5.3 Checkpoints — never in Git

A DeBERTa-v3-base checkpoint stores the full model state, so each
`best_model.pt` is roughly 700 MB – 1 GB. With five conditions plus five
ablations, and three checkpoint files per run, the total runs to tens of
gigabytes.

GitHub's hard limits: 100 MB per file, and repositories above ~5 GB draw
warnings. Git LFS on the free tier gives 1 GB of storage and 1 GB/month of
bandwidth — not enough, and it fails in a way that is painful to undo.

Publish weights instead as either:

- **A GitHub Release.** Up to 2 GB per attached file, unlimited files, no LFS
  quota. Good for one or two representative checkpoints.
- **Zenodo.** 50 GB per record, and it mints a DOI you can cite. This is the
  right home for a full checkpoint set, and most journals prefer a DOI to a
  GitHub link.

Upload `model_only.safetensors` rather than `best_model.pt` when you only need
to let someone reproduce inference — it is roughly a third of the size and
carries no optimizer state.

Always ship `calibration/temperature.npy` and `calibration/threshold.npy`
alongside any checkpoint. Without them the model cannot reproduce the reported
decisions, because the threshold was selected on validation and frozen.

### 5.4 Data — check the licence first

MedHallu and PubMedQA carry their own terms. Do not re-host the derived splits
until you have confirmed redistribution is permitted. If it is not, publish the
row IDs and a reconstruction script instead, so a reader can rebuild the exact
splits from the original source.

Whatever you publish, include the SHA-256 hashes from `run_manifest.json`.
That is what lets someone prove they rebuilt the same splits.

### 5.5 Never publish

- `OPENROUTER_API_KEY`, in any file, including notebook output cells.
- `data/cache/` and `artifacts/baselines/*/api_cache/` — raw API responses may
  contain provider metadata and account identifiers.
- `.env`.

Before the first push:

```bash
grep -rn "sk-or-v1" . --exclude-dir=.git
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```

The second command strips notebook outputs, which is where keys most often leak.

---

## 6. Publishing to GitHub

### 6.1 First push

```bash
cd clinhallu
git init -b main
git add .
git status                      # read this list carefully before committing
git commit -m "ClinHallu 2.2.1: GAER++ medical hallucination detection"
```

Create an **empty** repository on github.com — no README, no .gitignore, no
licence, or the first push will be rejected for unrelated histories. Then:

```bash
git remote add origin https://github.com/<username>/clinhallu.git
git push -u origin main
```

If prompted for a password, GitHub wants a personal access token, not your
account password: Settings → Developer settings → Personal access tokens →
Fine-grained tokens, with Contents: read and write scope.

### 6.2 Verify before pushing

`git status` after `git add .` should show roughly 200 files and no `.pt`,
`.jsonl`, `.npy` or `.env`. If it shows any of those, stop and fix `.gitignore`
rather than pushing and deleting afterwards — a deleted file stays in Git
history and remains publicly recoverable.

```bash
git count-objects -vH | grep size-pack     # expect a few MB
git ls-files | wc -l
git ls-files | grep -E '\.(pt|safetensors|jsonl|npy)$'   # expect no output
```

### 6.3 Attaching results to a release

```bash
git tag -a v2.2.1 -m "Paper submission"
git push origin v2.2.1
```

Then on GitHub: Releases → Draft a new release → pick the tag → attach
`model_only.safetensors`, `temperature.npy`, `threshold.npy` and a zipped
`artifacts/reports/`. Files up to 2 GB each.

### 6.4 Getting a DOI

Link Zenodo to your GitHub account, enable the repository, then publish a
release. Zenodo archives the tagged snapshot and issues a DOI automatically.
Put that DOI in the paper's data-availability statement and in `CITATION.cff`.

### 6.5 Before you make it public

- Update `CITATION.cff` — it currently reads `LicenseRef-Proprietary`, which
  conflicts with a public release. Decide on the real licence.
- `LICENSE` is proprietary too. Most venues expect MIT or Apache-2.0 for code.
- Add the paper title, arXiv link and DOI to the top of `README.md`.
- Consider keeping the repository private until acceptance, then flipping it
  public. Reviewers can be given an anonymised link via anonymous.4open.science.

---

## 7. Pre-submission checklist

Reproducibility

- [ ] `python -m pytest` passes
- [ ] `python scripts/check_environment.py` recorded with GPU and library versions
- [ ] All five conditions evaluated, `run_manifest.json` present for each
- [ ] All five ablations evaluated
- [ ] `scripts/verify_results.py` confirms `shared_frozen_splits: true`
- [ ] Seeds stated in the paper and matching the run directories

Protocol

- [ ] Threshold and temperature selected on validation only, never on test
- [ ] No question/context group appears in more than one split
- [ ] Dataset SHA-256 hashes recorded and reported
- [ ] Baselines evaluated on the identical frozen split as the model

Reporting

- [ ] Metrics reported as fractions or converted consistently
- [ ] Bootstrap confidence intervals included for the headline F1
- [ ] SelfCheckGPT described as a context-conditioned SelfCheckGPT-NLI
      adaptation, as `README.md` §11 explains
- [ ] Judge probabilities described as self-reported, not calibrated

Repository

- [ ] No API keys anywhere, notebook outputs cleared
- [ ] No checkpoints or `.jsonl` data committed
- [ ] Licence in `LICENSE` and `CITATION.cff` agrees with what you intend
- [ ] README links to the data archive and the checkpoint DOI

---

## 8. Known limits

`configs/base.yaml` pins `microsoft/deberta-v3-base` at revision
`559062ad13d311b87b2c455e67dcd5f1c8f65111`. Keep that pin in the paper's
reproducibility section; a silent upstream update would otherwise change results.

The `--smoke-test` flag deliberately skips frozen-test evaluation. Never report
a smoke-test number.

Re-running `evaluate --force` recomputes metrics from the existing checkpoint
and calibration. It does not re-select anything on test labels, so it is safe,
but it will not fix a run whose calibration was built on the wrong split.
