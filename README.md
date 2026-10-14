# ClinHallu: complete training and baseline pipeline

[![Python](https://img.shields.io/badge/Python-3.10--3.13-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.57.6-ffcc4d)](https://huggingface.co/docs/transformers)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-LLM%20judges-6467f2?logo=openai&logoColor=white)](https://openrouter.ai/)

Code for the paper **"ClinHallu: Measuring Domain Shift and the Over-Flagging Bias in Medical Hallucination Detection"** (Umair Ali, Mamoona Tasadduq, Javed Ali Khan).
**Datasets**

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-PubMedQA-ffcc4d)](https://huggingface.co/datasets/qiaojin/PubMedQA)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-MedHallu-ffcc4d)](https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu)

ClinHallu is a controlled protocol that measures what a medical hallucination detector loses when its training data (automatically generated, PubMedQA-derived) and its evaluation data (curated, MedHallu-derived) come from different construction processes. Five nested training conditions (C1–C5) share one validation set, one frozen expert test set, the same calibration procedure, and the same seeds.

This project trains GAER++, selects the checkpoint on validation data, calibrates it on validation data, and evaluates on a frozen test split. It also runs five independently trained ablations (B–F) and provides nine configured baseline instances. The paper reports every GAER++ result as a mean over seeds **13, 42, 77**.

**The consistency baselines share the same five answers.** For official-style SelfCheckGPT sampling, five separate stochastic OpenRouter requests produce five sampled passages for each unique question/context pair. The answers are saved in a JSON array and reused locally. Neither scoring command generates answers or calls OpenRouter.

**Every long-running command resumes automatically.** If training, an ablation, a baseline, or the pipeline is interrupted — a Colab disconnect, a dropped network, a closed laptop, `Ctrl-C` — re-run the identical command and it continues from where it stopped rather than starting over. Training resumes *inside* the epoch it was in, not just at the last epoch boundary. See [resume.md](docs/resume.md).

There are no research datasets, API keys, or trained research checkpoints in this ZIP. Benchmark results are reported in the paper (summary in [Section 15](#15-results-reported-in-the-paper)), not shipped as files here. See [validation_report.md](docs/validation_report.md) for the executed local checks and their limits.

## 1. Install

Extract the archive and run commands from the `clinhallu` directory containing this README and `pyproject.toml`.

### Google Colab

Select **Runtime → Change runtime type → GPU**. Keep the project on Google Drive so checkpoints and paid-response caches survive a Colab reset. A T4 is a practical starting point (the paper's runs used a Tesla T4); reduce batch size if memory is insufficient.

Run this Python cell, after putting the extracted `clinhallu` folder in `MyDrive`:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/clinhallu
%pip install -r requirements/colab.txt
%pip install -e . --no-deps
```

Keep Colab's existing PyTorch/CUDA installation. Restart the runtime if pip requests it, then mount Drive and change directory again. The supplied dependency stack uses Transformers 4.57.6 and PEFT 0.18.1. The notebook [run_complete_pipeline.ipynb](notebooks/run_complete_pipeline.ipynb) includes setup and execution cells.

> **Environment note.** The paper's development environment is listed as Python 3.13, CUDA 12.8, PyTorch 2.11, Transformers 5.16, PEFT 0.20, Accelerate 1.14 and NumPy 2.5 (Appendix C). The pinned requirements above are the tested install stack for this repository. Exact numbers can shift slightly between library versions, so record your versions when comparing against the paper.

All shell commands below work in a terminal. In a Colab code cell, put `!` before each command, or put `%%bash` on the first line of a cell containing multiple shell commands.

### Local Python

Use Python 3.10–3.13 and install PyTorch for your CPU/CUDA environment. Then:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python scripts/check_environment.py
```

`python -m clinhallu` can replace `clinhallu` in every command below.

## 2. Put your fixed data splits in place

### Source datasets

The splits used in this work are derived from two public Hugging Face datasets:

| | Dataset | Role in this project |
|---|---|---|
| [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-PubMedQA-ffcc4d)](https://huggingface.co/datasets/qiaojin/PubMedQA) | **PubMedQA** | Biomedical QA built from PubMed abstracts (question, abstract context, yes/no/maybe answer). It is the upstream corpus for **both** pools: the 10,000-example **source** pool of automatically generated, PubMedQA-derived hallucination examples, and (through MedHallu) the target pool. |
| [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-MedHallu-ffcc4d)](https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu) | **MedHallu** | Medical hallucination-detection benchmark built over PubMedQA, with supported and hallucinated answers and `hallu_label`. Supplies the **target** pool: the four 1,000-example training increments (C2–C5), the 4,000-example validation set, and the 1,000-example expert test set. |

The source and target pools share a clinical domain and an upstream corpus. What differs is the construction process: corruption strategy, question phrasing, evidence length and annotation policy. The paper therefore calls the effect *source-to-benchmark shift* rather than domain shift. "Expert" denotes the provenance claimed by the upstream benchmark; annotator qualifications were not verified.

Neither dataset ships with this repository. Download them yourself, build the splits described below, and check their licence terms before redistributing any derived file (both upstream projects are MIT-licensed; derived files must restore attribution, upstream versions and a record of the transformation steps).

The default C5 experiment, baselines, and shared generation use:

```text
data/raw/train.jsonl
data/raw/val.jsonl
data/raw/eval_data.jsonl
```

Each line is a JSON object:

```json
{"id":"example_001","question":"What happened?","context":"The patient improved after treatment.","answer":"The patient improved.","hallu_label":0}
```

Required fields: `id` (string or integer), `question`, `context`, `answer` (nonempty strings), and `hallu_label` (0 = supported; 1 = hallucinated). The paper's records carry only these five fields; no category, difficulty or annotator metadata, so no error-type breakdown is possible. Optional metadata such as `group_id`, `difficulty`, `generation_model`, and `category` are preserved where supported if you have them. IDs must be unique within a split. Question/context groups must not overlap across train, validation, and evaluation. Repeated questions inside a training split (a question with several candidate answers) are allowed and are treated as groups. Validation and evaluation contain no repeated questions.

Context is answer-aware: up to eight sentences and roughly 280 context tokens are kept, scored by question similarity (0.45), answer similarity (0.45) and entity overlap (0.10).

### Splits and conditions

C5 expects **14,000 train / 4,000 validation / 1,000 evaluation** rows. All splits are class-balanced (positive rate 0.500); the expert test set is 500 supported / 500 hallucinated. Set the audited counts in `configs/base.yaml` before starting if your dataset differs. For example, a verified 13,999-row training file needs `expected_rows.train: 13999`; do not pad it with duplicates. Each split must contain both classes. All runs save dataset hashes and reject changes to an already registered protocol.

The five conditions are **exact cumulative supersets**. Each step keeps every earlier record and adds exactly 1,000 balanced MedHallu-derived records:

| Condition | Source (PubMedQA-derived) | Target (MedHallu-derived) | Total train |
|---|---|---|---|
| C1 | 10,000 | 0 | 10,000 |
| C2 | 10,000 | 1,000 | 11,000 |
| C3 | 10,000 | 2,000 | 12,000 |
| C4 | 10,000 | 3,000 | 13,000 |
| C5 | 10,000 | 4,000 | 14,000 |
| Validation (shared) | 0 | 4,000 | 4,000 |
| Expert test (shared) | 0 | 1,000 | 1,000 |

Their configs do **not** construct subsets automatically. Each condition reads its own fixed training file, and all five share one held-out validation and evaluation split:

```text
data/raw/c1/train.jsonl      10,000 rows
data/raw/c2/train.jsonl      11,000 rows
data/raw/c3/train.jsonl      12,000 rows
data/raw/c4/train.jsonl      13,000 rows
data/raw/train.jsonl         14,000 rows   (C5)
data/raw/val.jsonl            4,000 rows   (shared)
data/raw/eval_data.jsonl      1,000 rows   (shared)
```

Adjust `data.expected_rows.train` in each `configs/conditions/cN.yaml` if your audited counts differ. C5 deliberately reads `data/raw/train.jsonl` rather than a `c5/` subdirectory, because the ablation configs inherit their training path from `configs/base.yaml`. Moving C5's data without also editing `configs/base.yaml` makes `clinhallu ablate` stop with *"Data registered as 'c5' changed"*.

Because validation is drawn from the target distribution in **every** condition (including C1), each condition receives its own best temperature and threshold. Differences between conditions therefore reflect the training mixture alone.

Validate C5:

```bash
clinhallu validate --config configs/conditions/c5.yaml
```

If you change data locations, update the model config, `configs/baselines/suite.yaml`, and `configs/generation/shared_five.yaml` consistently.

## 3. Set your OpenRouter API key

The code reads `OPENROUTER_API_KEY` from the process environment. It does not load `.env` automatically. Never paste a real key into a YAML or prompt file.

In Colab, add a secret named `OPENROUTER_API_KEY`, enable notebook access, and run:

```python
import os
from google.colab import userdata
os.environ['OPENROUTER_API_KEY'] = userdata.get('OPENROUTER_API_KEY')
```

Alternatively, in a Python notebook:

```python
import getpass
import os

os.environ['OPENROUTER_API_KEY'] = getpass.getpass('OpenRouter API key: ')
```

In Bash:

```bash
read -rs -p 'OpenRouter API key: ' OPENROUTER_API_KEY
export OPENROUTER_API_KEY
```

The default generator is `openai/gpt-4.1-mini`. Change `model_name` in `configs/generation/shared_five.yaml` to use another OpenRouter model. The judge instances are:

Four OpenRouter models are used as zero-shot LLM judges:

| | Judge | Suite name | OpenRouter model ID | Notes |
|---|---|---|---|---|
| [![OpenRouter](https://img.shields.io/badge/OpenRouter-GPT--5%20Mini-10a37f?logo=openai&logoColor=white)](https://openrouter.ai/openai/gpt-5-mini) | **GPT-5 Mini** | `openrouter_gpt_5_mini` | `openai/gpt-5-mini` | Minimal reasoning, 4096-token completion budget, no custom temperature. |
| [![OpenRouter](https://img.shields.io/badge/OpenRouter-DeepSeek%20V3-4d6bfe?logo=deepseek&logoColor=white)](https://openrouter.ai/deepseek/deepseek-chat) | **DeepSeek V3.2 (Chat)** | `openrouter_deepseek_chat` | `deepseek/deepseek-chat` | The paper reports this judge as DeepSeek-V3.2. The `deepseek-chat` alias can move between versions, so check what it resolves to (`clinhallu api models --contains deepseek`) and record the resolved model in your results. |
| [![OpenRouter](https://img.shields.io/badge/OpenRouter-Llama%203.3%2070B-0866ff?logo=meta&logoColor=white)](https://openrouter.ai/meta-llama/llama-3.3-70b-instruct) | **Llama 3.3 70B Instruct** | `openrouter_llama_3_3_70b_instruct` | `meta-llama/llama-3.3-70b-instruct` | Meta multilingual instruction-tuned 70B model. |
| [![OpenRouter](https://img.shields.io/badge/OpenRouter-Qwen3%2032B-6f42c1?logo=alibabacloud&logoColor=white)](https://openrouter.ai/qwen/qwen3-32b) | **Qwen3 32B** | `openrouter_qwen3_32b` | `qwen/qwen3-32b` | Reasoning ("thinking mode") disabled for the judge task. |

Config files live under `configs/baselines/llm_judges/<suite name>.yaml`. Each judge has its own output and API-cache directory. Reports keep every model separate. All four judges receive an identical system prompt stating the annotation definition, restricting the model to the supplied context, and requiring a JSON object with a binary label and a hallucination probability.

Inspect the current public catalog without a paid completion:

```bash
clinhallu api models --contains gpt-5-mini
clinhallu api models --contains deepseek-chat
```

API parameters and endpoint compatibility are documented in [api_sources.md](docs/api_sources.md). Model availability can change, and OpenRouter may route a request to different serving backends within one run, so judge numbers describe a routed service rather than one fixed deployment and may not reproduce exactly. Judge responses are validated locally. Harmless provider-added JSON fields are recorded and ignored; missing or invalid required values are rejected. GPT-5 Mini omits the unsupported custom temperature parameter, uses minimal reasoning, and has a 4096-token completion budget because its reasoning tokens count against that limit. Qwen3 32B disables optional reasoning for its short JSON response budget.

## 4. Run the whole pipeline

Preview the commands (no training, data requests, or paid completions):

```bash
clinhallu pipeline --condition c5 --seed 13 --dry-run
```

Run everything:

```bash
clinhallu pipeline --condition c5 --seed 13
```

The paper reports three seeds. Repeat the pipeline per seed (each seed gets its own directory):

```bash
for seed in 13 42 77; do
  clinhallu pipeline --condition c5 --seed $seed
done
```

The shared answer generation and the baselines are seed-independent, so they only need to run once; later seeds skip them as completed stages (use `--stages model ablations report` to be explicit).

The stages run sequentially:

1. Prepare C5, train GAER++, calibrate on validation, evaluate the frozen split.
2. Independently train and evaluate ablations B, C, D, E, and F.
3. Generate five independently sampled answers per evaluation question/context.
4. Run the five default-enabled baselines: three conventional methods, SelfCheckGPT-NLI, and self-consistency. The four paid judges are run explicitly one at a time.
5. Build tables, summaries, and figures.

Each stage is recorded as it completes, so re-running this command after an interruption skips the stages that finished and picks up the first one that did not. The stage that was cut off resumes from its own checkpoint rather than from its beginning. Cached API completions are reused as before. Errors stop the relevant stage; fix the reported cause and rerun the same command. No training or validation data are sent to OpenRouter by default.

Check what a re-run would still have to do:

```bash
clinhallu status
clinhallu pipeline --condition c5 --seed 13 --status
```

To force a finished stage to run again:

```bash
clinhallu pipeline --condition c5 --seed 13 --force-stage baselines
clinhallu pipeline --condition c5 --seed 13 --no-resume   # every stage again
```

To run selected stages:

```bash
clinhallu pipeline --condition c5 --seed 13 --stages model report
clinhallu pipeline --stages generate baselines report
```

The supplied ablation configs use C5. For C1–C4, omit the `ablations` stage or use the individual condition commands below.

## 5. Run training and evaluation individually

Prepare only:

```bash
clinhallu prepare --config configs/conditions/c5.yaml --seed 13
```

Train and perform validation calibration:

```bash
clinhallu train --config configs/conditions/c5.yaml --seed 13
```

Resume an interrupted training run from its latest epoch checkpoint:

```bash
clinhallu train --config configs/conditions/c5.yaml --seed 13 --resume
```

Evaluate using the best checkpoint and saved validation temperature/threshold:

```bash
clinhallu evaluate --config configs/conditions/c5.yaml --seed 13
```

Explicitly recompute an existing evaluation without selecting anything on test labels:

```bash
clinhallu evaluate --config configs/conditions/c5.yaml --seed 13 --force
```

Prepare, train, calibrate, and evaluate one condition with automatic resume:

```bash
clinhallu condition c5 --seed 13
```

Other conditions, once their training files are in place and their audited row counts are set:

```bash
clinhallu condition c1 --seed 13
clinhallu condition c2 --seed 13
clinhallu condition c3 --seed 13
clinhallu condition c4 --seed 13
```

To reproduce the full C1–C5 adaptation curve over the paper's three seeds:

```bash
for seed in 13 42 77; do
  for c in c1 c2 c3 c4 c5; do
    clinhallu condition $c --seed $seed
  done
done
```

Run a short training smoke check (it does not evaluate the frozen test split):

```bash
clinhallu condition c5 --seed 13 --smoke-test
```

For another seed, change `--seed`, for example `--seed 42`. Every seed gets a separate directory.

All conditions share preprocessing and hyperparameters (AdamW, cosine decay with 10% warmup, at most five epochs with patience two, batch size 8 × 4 accumulation; full table in the paper's Appendix C). Only the training data change. Test labels never influence the temperature `T` or threshold `τ`: `T` minimises validation cross-entropy and `τ` is swept on validation data (0.05–0.95, step 0.01) to maximise hallucinated-class F1.

Resume restores model, optimizer, scheduler, scaler, epoch, early-stopping state, the position inside the epoch, the partially accumulated epoch metrics, and the RNG state of Python, NumPy, PyTorch CPU and PyTorch CUDA. Because the training order for epoch *k* is derived from `(seed, k)` rather than from a generator that advances as the process runs, a resumed run continues the same trajectory: it produces bit-identical weights to an uninterrupted run. This is enforced by a test that trains an epoch twice, once straight through and once killed half way, and compares the weights exactly.

To deliberately discard saved progress and train from scratch:

```bash
clinhallu condition c5 --seed 13 --no-resume
```

## 6. Run the ablations individually

The ablation is cumulative on C5; each rung adds one block:

| Variant | Features used by its classifier |
|---|---|
| B | Fused CLS/answer representation (attention pooling); evidence features zeroed |
| C | B plus answer-to-context cross-attention and uniform evidence pooling |
| D | C plus learned token-grounding weights for evidence pooling |
| E | D plus six grounding/pooling statistics |
| F | E plus answer/evidence difference, product, and cosine interactions (full GAER++) |

All variants jointly encode question, context, and answer. B disables the explicit evidence branch; it does not remove context from the encoder input. Evidence pooling already exists in C/D; E exposes its statistics. F is trained separately, even when the C5 full model has already run.

In the paper's three-seed results, the whole evidence pathway (F − B) is worth only +0.24 F1 (within seed noise) but +0.39 AUROC and +0.72 MCC. Rung E (the six grounding statistics) lowers F1 by 1.69 and is the only negative step. The paper recommends omitting that block, although the released F configuration includes it because it is the configuration evaluated throughout.

Running all five at once is safe to interrupt. Each variant's outcome is recorded the moment it changes, so re-running the same command skips the variants that finished, resumes the one that was in flight from its own training checkpoint, and continues with the rest:

```bash
clinhallu ablate --variants B C D E F --seed 13   # stops during D
clinhallu ablate --variants B C D E F --seed 13   # skips B and C, continues D mid-epoch
```

Add `--continue-on-error` to keep going past a variant that fails outright (for example on an out-of-memory error), so one bad variant does not block the other four. Failures are still reported at the end and are retried on the next run.

```bash
clinhallu ablate --variants B --seed 13
```

```bash
clinhallu ablate --variants C --seed 13
```

```bash
clinhallu ablate --variants D --seed 13
```

```bash
clinhallu ablate --variants E --seed 13
```

```bash
clinhallu ablate --variants F --seed 13
```

Run all five (repeat with `--seed 42` and `--seed 77` for the three-seed results):

```bash
clinhallu ablate --variants B C D E F --seed 13
```

## 7. Run conventional baselines individually

TF-IDF + logistic regression (trained on the same 14,000-example C5 split as GAER++, threshold selected on validation):

```bash
clinhallu baseline run --only tfidf_lr
```

Biomedical semantic similarity (PubMed sentence encoder; scores answer claims against context windows and takes the weakest-supported claim):

```bash
clinhallu baseline run --only semantic_similarity
```

Zero-shot NLI (DeBERTa cross-encoder; answers are split into claims and scored against context windows, with contradiction weight set to zero, i.e. an entailment-deficit detector):

```bash
clinhallu baseline run --only zero_shot_nli
```

The semantic and NLI models run locally and may download Hugging Face weights on first use. Their thresholds are selected using validation only. Neither calls OpenRouter.

## 8. Run each OpenRouter LLM judge

GPT-5 Mini:

```bash
clinhallu baseline run --only openrouter_gpt_5_mini
```

This writes to `artifacts/baselines/openrouter_gpt_5_mini_v3`. It intentionally
does not reuse the older `v2` cache whose 512-token GPT-5 requests can end with
`finish_reason: length` before the JSON answer is emitted.

DeepSeek V3.2 (Chat):

```bash
clinhallu baseline run --only openrouter_deepseek_chat
```

Llama 3.3 70B Instruct:

```bash
clinhallu baseline run --only openrouter_llama_3_3_70b_instruct
```

Qwen3 32B:

```bash
clinhallu baseline run --only openrouter_qwen3_32b
```

Choose a model directly, with a separate output directory:

```bash
python -m clinhallu.baselines.llm_judge \
  --provider openrouter \
  --model-name openai/gpt-5-mini \
  --api-config configs/api/judge_gpt5.yaml \
  --train data/raw/train.jsonl \
  --val data/raw/val.jsonl \
  --eval data/raw/eval_data.jsonl \
  --output-dir artifacts/baselines/my_judge
```

Add `--dry-run` to that direct command to inspect remaining request count without calling the API. `clinhallu baseline run --dry-run` only prints the suite's planned commands.

Each judge receives question, context, and candidate answer from the evaluation split. Gold labels, training examples, and validation examples are never included in its prompt. It returns `hallu_label` and `probability_hallucinated`; the fixed decision rule is probability ≥ 0.5. These are self-reported probabilities, not guaranteed calibrated probabilities (in the paper they are left unscaled, so their calibration is not directly comparable to GAER++'s temperature-scaled probabilities). Predictions, usage, and actual model/provider metadata are saved.

To evaluate existing externally produced binary judgments:

```bash
python -m clinhallu.baselines.llm_judge \
  --provider cached_predictions \
  --model-name my_external_judge \
  --predictions path/to/judge_predictions.json \
  --train data/raw/train.jsonl --val data/raw/val.jsonl --eval data/raw/eval_data.jsonl \
  --output-dir artifacts/baselines/my_external_judge
```

The file must contain exactly matching evaluation IDs and binary `hallu_label` values. Binary-only imports do not produce probability calibration metrics.

## 9. Generate the five shared answers ONCE

The external prompt files are:

- `prompts/shared_five_system.txt`
- `prompts/shared_five_user.txt`

Their paths are selected by `configs/generation/shared_five.yaml`. Edit those text files to change the prompt. The loader automatically replaces the literal `{{INPUT_JSON}}` marker with a JSON object containing only `question` and `context`. The candidate answer and gold label are excluded. IDs and questions are copied locally into the export.

Count remaining requests first:

```bash
clinhallu generate --config configs/generation/shared_five.yaml --dry-run
```

Generate evaluation answers:

```bash
clinhallu generate --config configs/generation/shared_five.yaml
```

This command makes **five independent completions per unique question/context pair** using temperature 1.0 and seeds 13–17. For 1,000 unique evaluation pairs, a fresh cache requires 5,000 successful completions. It does not send the full dataset in one request.

Output: `data/generated/shared_five/evaluation_answers.json`, a JSON **array**, with exactly your requested per-row structure:

```json
[
  {
    "id": "example_001",
    "question": "What happened?",
    "answer_01": "The patient improved after treatment.",
    "answer_02": "After treatment, the patient improved.",
    "answer_03": "The context reports improvement following treatment.",
    "answer_04": "The patient showed improvement after receiving treatment.",
    "answer_05": "The reported outcome was improvement after treatment."
  }
]
```

The five answers may be similar or identical; the prompt never forces fabricated diversity. The original ID type is retained. A separate `generation_manifest.json` records model, prompt, request settings, input hashes and output hashes without adding metadata fields to the answer rows.

To limit a run to 100 new successful generation completions:

```bash
clinhallu generate --config configs/generation/shared_five.yaml --max-new-requests 100
```

If the limit is reached, the command exits with code 2 and status `incomplete`. Each completed API response is already cached, but an incomplete split is not exported as a complete answer array. Resume by running the same command again, or remove the limit to finish all remaining rows.

### Optional validation answers

The default fixed-threshold consistency baselines need only evaluation answers. Generate validation answers only if you elect to tune the SelfCheckGPT threshold on validation:

```bash
clinhallu generate --config configs/generation/shared_five.yaml --splits validation
```

Or generate both splits in one command:

```bash
clinhallu generate --config configs/generation/shared_five.yaml --splits validation test
```

Validation output: `data/generated/shared_five/validation_answers.json`. Cached evaluation responses are reused.

## 10. Run SelfCheckGPT with the saved answers

```bash
clinhallu baseline run --only selfcheckgpt
```

This implementation calls the official package's recommended `SelfCheckNLI`. It compares each candidate-answer sentence with all five sampled passages and returns sentence-level non-factuality scores, averaged over samples and then sentences into a document score. Larger scores mean greater hallucination risk. The default decision threshold is fixed at 0.5 (as in the paper) and needs no validation generation.

Optional validation-selected threshold, after generating validation answers:

```bash
python -m clinhallu.baselines.selfcheckgpt \
  --train data/raw/train.jsonl \
  --val data/raw/val.jsonl \
  --eval data/raw/eval_data.jsonl \
  --answers-dir data/generated/shared_five \
  --threshold-mode validation \
  --output-dir artifacts/baselines/selfcheckgpt_validation_tuned
```

For the suite to use this protocol, set `threshold_mode: validation` and a new `output_dir` in `configs/baselines/selfcheckgpt.yaml`. Do not select between thresholds using evaluation results.

## 11. Run self-consistency with the SAME saved answers

> This baseline is provided by the code but is **not** part of the paper's Table 6; the paper's sampling-consistency detector is SelfCheckGPT-NLI (Section 10).

```bash
clinhallu baseline run --only self_consistency
```

The local NLI model gives each sampled answer one support vote: it must entail every candidate sentence with probability ≥ 0.5. The hallucination score is `1 - support_votes / 5`. At least three non-support votes predict hallucination; at least three support votes predict supported. The decision rule is fixed before evaluation.

Run both scoring methods together after generation:

```bash
clinhallu baseline run --only selfcheckgpt self_consistency
```

Both read `data/generated/shared_five/evaluation_answers.json`. They require **zero additional answer-generation API calls** after sampling. GPU/CPU scoring has local compute cost. No separate API scoring prompt is needed; [prompts/README.md](prompts/README.md) explains which prompt each stage uses.

**Method naming:** SelfCheckGPT uses the official NLI scorer and independent samples, but conditions sampling on ClinHallu's supplied context; report it as a context-conditioned SelfCheckGPT-NLI adaptation. If the benchmark candidate came from a different generator, also report it as cross-model consistency. Agreement is a consistency signal, not proof of factual correctness.

## 12. Run all baselines and build reports

Generate shared evaluation answers first, then:

```bash
clinhallu baseline list
clinhallu baseline run --dry-run
clinhallu baseline run
clinhallu report build
```

`configs/baselines/suite.yaml` contains nine configured entries. Five local/scoring entries are enabled by default; all four paid judges are disabled by default to prevent accidental bulk spending. Run a judge explicitly with `--only`; selected entries run even when disabled by default.

Rebuild only baseline reports:

```bash
clinhallu baseline summarize
```

| Output | Location |
|---|---|
| C5 checkpoints, calibration, predictions, metrics | `artifacts/adaptation/c5/seed_13/` |
| Independently trained B–F results | `artifacts/ablations/<variant>/seed_13/` |
| Baseline aligned predictions and metrics | `artifacts/baselines/<instance>/` |
| Shared five-answer JSON files and generation manifest | `data/generated/shared_five/` |
| Raw generation API response cache | `data/cache/openrouter/shared_five/` |
| Raw judge API response cache | `artifacts/baselines/<judge>/api_cache/` |
| Shared local NLI cache | `data/cache/baselines/shared_nli/` |
| Publication tables | `artifacts/reports/tables/` |
| Figures (PNG/PDF) | `artifacts/reports/figures/` |
| Markdown summaries | `artifacts/reports/summaries/` |

Metrics use fractions, not percentages (the paper tables show percentages; MCC is on a [−100, 100] scale there). Metrics include accuracy, hallucinated-class precision/recall/F1, macro-F1, MCC, AUROC and AUPRC where defined. Brier/ECE (raw and temperature-scaled) are reported for probability outputs; normalized consistency/similarity scores and label-only imports omit them. AUPRC uses average precision. The neural model also reports grouped bootstrap F1 intervals (10,000 replicates; on the expert test set this coincides with ordinary resampling because every item has a distinct question). Comparisons must use the same frozen evaluation data and declared protocol.

Threshold protocol differs by method, so compare with care: GAER++, TF-IDF, semantic similarity and zero-shot NLI use validation-selected thresholds, whereas SelfCheckGPT-NLI and the four judges decide at a fixed cut (0.5). AUROC and AUPRC are independent of that choice.

## 13. Caches, changes, and cost control

- Save the project, answer directory, raw API caches, and NLI caches together. Generation and judge requests are cached by their full request body. Identical question/context pairs share generated answers even when candidate IDs differ.
- Raw completions are written before parsing and before the final array export. Malformed JSON, a refusal, truncation, missing fields, duplicate fields, or inconsistent judge label/probability stops the run. These do not trigger automatic paid regeneration or a fallback label.
- Retry count defaults to **zero**. You can explicitly set `request.rate_limit_retries` to allow bounded retries on HTTP 429 only. Transport errors/timeouts may have incurred a charge and are not automatically retried.
- Re-running a command with a cached malformed response fails at that same response without another call. Inspect the cache path in the error. If you intentionally want to retry, remove that failed cache entry. For changed prompt/model/token settings, use a new output directory; the changed request gets a different cache key and may cost money.
- Data, prompt, generation model, or scoring protocol changes require a new experiment/output directory. This prevents mixing results from different protocols. Never hand-edit a generated answer file while retaining its old manifest.
- All five sampled answers are generated by five independent completions. The completed answer file is shared by SelfCheckGPT and self-consistency, avoiding duplicate generation across those two baselines. Usage files record token counts and provider-reported cost when available; a missing cost is explicitly counted rather than assumed zero.
- Run one writer per output/cache directory. Concurrent independent runs should use different directories.

## 14. Verify the installation

```bash
python -m pytest
```

To watch the resume machinery work on a real (tiny) training run — interrupted
by an actual signal, then restarted — without a GPU or network:

```bash
python scripts/demo_resume.py
```

The test suite includes actual training/calibration/evaluation with a tiny locally created DeBERTa model, every B–F ablation, local NLI and sentence-embedding runtime checks, shared-answer cache tests, and mocked OpenRouter transport. These are implementation checks, not medical benchmark results. The original research dataset, full pretrained model/GPU experiments, and live paid API completions still need to be run in your environment.

## 15. Results reported in the paper

Frozen 1,000-example expert test set, mean ± SD over seeds 13/42/77 (%). These come from the paper, not from this repository's tests.

| Condition | F1 | MCC | AUROC |
|---|---|---|---|
| C1 (source only) | 77.11 ± 1.90 | 51.80 ± 4.79 | 83.82 ± 2.24 |
| C2 (+1k target) | 90.38 ± 0.50 | 80.88 ± 0.95 | 96.76 ± 0.04 |
| C3 (+2k target) | 92.08 ± 0.49 | 84.64 ± 1.03 | 97.75 ± 0.18 |
| C4 (+3k target) | 92.77 ± 0.25 | 85.97 ± 0.70 | 98.10 ± 0.09 |
| C5 (+4k target) | 93.75 ± 0.10 | 87.71 ± 0.35 | 98.49 ± 0.21 |

Main findings: source-only training costs 16.64 F1 and 35.91 MCC; ranking survives better than decision quality; the first 1,000 target examples recover 79.7% of the F1 gain; every zero-shot system and the source-only encoder over-flags (recall > precision), and only target adaptation reverses it; at 5% prevalence the adapted detector's PPV drops to 50.3%. The strongest baseline is Llama-3.3-70B at 89.30 F1.

Caveats stated in the paper: condition is confounded with training-set size (no size-matched source-only control was run); no target-only condition was trained; GAER++ was not run with blanked or shuffled context, and a TF-IDF baseline reaches 71.06 F1, so surface regularities bound how much any score reflects evidence reasoning; and "expert" labels reflect upstream provenance, not verified annotator qualifications. Results are benchmark-relative and are not evidence of clinical readiness.

Preparing a paper submission or publishing this repository: [submission guide](SUBMISSION_GUIDE.md).

Further documentation: [resuming interrupted runs](docs/resume.md), [architecture](docs/architecture.md), [data protocol](docs/data_protocol.md), [baseline protocol](docs/baseline_protocol.md), [shared-answer protocol](docs/shared_answer_protocol.md), [2.2 changes](docs/changes_2_2.md), and [validation report](docs/validation_report.md).
