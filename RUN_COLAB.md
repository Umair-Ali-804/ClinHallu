# ClinHallu 2.2.1: exact Colab run order

Run every command from `/content/drive/MyDrive/clinhallu`.

## 1. Install the verified dependency stack

```python
from google.colab import drive

drive.mount("/content/drive")
%cd /content/drive/MyDrive/clinhallu
%pip install -r requirements/colab.txt
%pip install -e . --no-deps
```

Restart the runtime if Colab requests it. Mount Drive and change to the project
directory again after restarting. Do not run the project with Transformers 5.x;
the supplied requirements install the verified Transformers 4.57.6 and PEFT
0.18.1 combination.

## 2. Check the installation and data

Place these files under `data/raw/`:

- `train.jsonl`
- `val.jsonl`
- `eval_data.jsonl`

Then run:

```bash
!python scripts/check_environment.py
!clinhallu validate --config configs/conditions/c5.yaml --seed 13
```

## 3. Run GAER++ C5

```bash
!clinhallu condition c5 --seed 13
```

The same command resumes after an interruption, including from the middle of an
epoch. Run it again after reconnecting and it continues from its last checkpoint.

## 3a. When the Colab session drops

This will happen: the twelve-hour limit, an idle timeout, a reclaimed GPU. It is
not a problem. Reconnect, then:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/clinhallu
%pip install -r requirements/colab.txt
%pip install -e . --no-deps
```

Check what survived, then re-run the identical command:

```bash
!clinhallu status
!clinhallu condition c5 --seed 13
```

Two things make this work. Checkpoints are written every 200 optimizer steps or
every 10 minutes, whichever comes first, so at most that much work is lost. And
the run directory lives on Drive, so it outlives the VM — which is why step 1
mounts Drive and `cd`s into it before anything else.

If checkpoint writes feel slow on Drive, widen the interval in
`configs/base.yaml`:

```yaml
checkpoints:
  save_every_steps: 500
  save_every_minutes: 20
```

Full details: [docs/resume.md](docs/resume.md).

## 4. Run ablations independently

```bash
!clinhallu ablate --variants B --seed 13
!clinhallu ablate --variants C --seed 13
!clinhallu ablate --variants D --seed 13
!clinhallu ablate --variants E --seed 13
!clinhallu ablate --variants F --seed 13
```

Or run all five in one command. Completed variants are recorded, so re-running
after a dropped session skips them and resumes the one that was in flight:

```bash
!clinhallu ablate --variants B C D E F --seed 13
```

## 5. Run conventional baselines

```bash
!clinhallu baseline run --only tfidf_lr
!clinhallu baseline run --only semantic_similarity
!clinhallu baseline run --only zero_shot_nli
```

## 6. Set the OpenRouter key

```python
import os
from getpass import getpass

os.environ["OPENROUTER_API_KEY"] = getpass("OpenRouter API key: ")
```

## 7. Run LLM judges one at a time

```bash
!clinhallu baseline run --only openrouter_gpt_5_mini
!clinhallu baseline run --only openrouter_gemini_2_5_flash
!clinhallu baseline run --only openrouter_deepseek_chat
!clinhallu baseline run --only openrouter_llama_3_3_70b_instruct
!clinhallu baseline run --only openrouter_qwen3_32b
!clinhallu baseline run --only openrouter_llama_4_maverick
!clinhallu baseline run --only openrouter_qwen3_30b_a3b
!clinhallu baseline run --only openrouter_gemini_2_5_flash_lite
```

GPT-5 Mini uses its new `openrouter_gpt_5_mini_v3` output directory, minimal
reasoning, and a 4096-token completion budget. Leave the old `v2` directory in
place; the command above will not reuse its truncated 512-token response.

Each successful response is cached immediately. Re-running the same command
continues from the first uncached evaluation record. Provider-added JSON fields
are safely ignored and recorded; the required label and probability remain
validated.

## 8. Generate five responses and run SelfCheckGPT

Preview the number of required generation requests:

```bash
!clinhallu generate --config configs/generation/shared_five.yaml --splits test --dry-run
```

Generate or resume five independent answers per evaluation record:

```bash
!clinhallu generate --config configs/generation/shared_five.yaml --splits test
```

Run the official-style local SelfCheckGPT-NLI scorer:

```bash
!clinhallu baseline run --only selfcheckgpt
```

Optionally run the separate self-consistency detector using the same generated
answer file:

```bash
!clinhallu baseline run --only self_consistency
```

The two scoring commands do not call OpenRouter.

## 9. Build results

```bash
!clinhallu baseline summarize
!clinhallu report build
!python scripts/verify_results.py artifacts/adaptation/c5/seed_13
```

## Important resume rule

Do not delete `artifacts/**/checkpoints/`, `stage_ledger.json`,
`ablation_ledger.json`, `baseline_ledger.json`, or `artifacts/pipeline/`. Those
are what let an interrupted run continue. If you genuinely want to start over,
use `--no-resume` rather than deleting files by hand.

## Important cache rule

Do not delete `api_cache` or `data/cache/openrouter/shared_five` after an
interrupted API run. Delete a specific cached response only when its error
message explicitly says that saved completion is malformed and you intentionally
want to pay for a replacement request.
