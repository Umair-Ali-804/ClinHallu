# Baseline protocol

Labels are fixed: supported=0 and hallucinated=1. Baselines validate all three raw splits, require both classes, check unique IDs and disjoint question/context groups, and save evaluation predictions in input order. Gold labels are used only for the stated training/threshold/evaluation roles and never placed in an API prompt.

| Method | Fit / threshold source | External completion scope |
|---|---|---|
| TF-IDF + logistic regression | Fit on the C5 training split (14,000 rows); threshold from validation | None |
| Biomedical semantic similarity | Pretrained local PubMed sentence encoder; threshold from validation | None |
| Zero-shot NLI | Pretrained local DeBERTa cross-encoder; threshold from validation | None |
| Four OpenRouter judges (GPT-5 Mini, DeepSeek V3.2, Llama 3.3 70B Instruct, Qwen3 32B) | Fixed prompt rule, probability ≥ 0.5 | Evaluation candidates only |
| Context-conditioned SelfCheckGPT-NLI | Fixed threshold 0.5 by default; optional validation F1 tuning | Five independent samples per item |
| Self-consistency batched detector | Fixed majority of five support votes | The same shared answer generation |

The suite has nine configured instances: three conventional baselines, four judges and two shared-answer detectors. The five local/scoring entries are enabled by default. Paid judges are disabled by default and run explicitly one at a time. Each judge has a distinct instance directory and model identifier.

The self-consistency detector is provided by the code but is not part of the paper's reported baselines. The paper's sampling-consistency baseline is SelfCheckGPT-NLI.

## Method details

- **TF-IDF + logistic regression** is included to expose lexical shortcuts. It sees the same training data as the main detector.
- **Semantic similarity** scores answer claims against context windows and takes the weakest-supported claim.
- **Zero-shot NLI** splits the answer into claims and scores each against context windows. Contradiction weight is set to zero, so it is an entailment-deficit detector.
- **Judges** all receive an identical system prompt that states the annotation definition, restricts the model to the supplied context, and requires a JSON object with a binary label and a hallucination probability.
- **SelfCheckGPT-NLI** scores each candidate sentence against each of the five samples, then averages over samples and then sentences.

## Generation

The default generation model is `openai/gpt-4.1-mini`. Five independent completions per exact question/context return five answers, saved in a JSON array with `id`, `question`, and `answer_01` through `answer_05`. Generation excludes the benchmark candidate and its label. Default generation is evaluation only; optional validation generation supports validation-selected SelfCheck thresholds.

SelfCheckGPT uses the official package's recommended `SelfCheckNLI` scorer. Sampling is independent but context-conditioned, so the paper must identify it as a context-conditioned adaptation.

## Judge responses and failures

Judges return an integer label and continuous self-reported hallucination probability. Provider-added fields are ignored and recorded; the two required values remain strictly validated. Missing, invalid, truncated, refused, or inconsistent responses stop the run, with raw completions retained for inspection and reuse. There is no fallback label. API client retries default to zero, and only explicit HTTP 429 retries are available. No validation or training API calls occur in the default suite.

Judge routing can change between serving backends within one run, so the actual returned model and provider metadata are saved with each experiment.

## Metrics and comparability

Probability baselines report Brier/ECE; normalized similarity and consistency scores do not. Judge AUROC/AUPRC use the probabilities the judges report, so they are comparable across rows. These probabilities are left unscaled, whereas the main detector's probabilities are temperature-scaled on validation, so calibration comparisons between them are not like-for-like. Imported binary-only judge predictions omit Brier/ECE and yield only coarse ranking information. AUROC/AUPRC for a binary score are not equivalent to ranking based on continuous confidence.

Thresholds differ by method: similarity, NLI and TF-IDF use validation-selected thresholds, while SelfCheckGPT-NLI and the judges use a fixed cut. AUROC and AUPRC do not depend on that choice, so compare methods on them first. Never choose model settings or thresholds using the frozen evaluation labels.

Each baseline in the paper is a single run, so no variance estimate accompanies it.

## Outputs

Model/config hashes, raw input hashes, thresholds, environment information, and aligned predictions are written under `artifacts/baselines/<instance>/`. Reports keep judge models separate. All baselines must be scored against the same frozen evaluation file (check the shared SHA-256 digest). Multi-seed comparisons should use the same fixed data and protocol; not every local baseline supports a suite-wide seed override.
