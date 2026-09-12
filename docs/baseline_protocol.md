# Baseline protocol

Labels are fixed: supported=0 and hallucinated=1. Baselines validate all three raw splits, require both classes, check unique IDs and disjoint question/context groups, and save evaluation predictions in input order. Gold labels are used only for the stated training/threshold/evaluation roles and never placed in an API prompt.

| Method | Fit / threshold source | External completion scope |
|---|---|---|
| TF-IDF + logistic regression | Fit on train; threshold from validation | None |
| Biomedical semantic similarity | Pretrained local model; threshold from validation | None |
| Zero-shot NLI | Pretrained local model; threshold from validation | None |
| Eight OpenRouter judges | Fixed prompt rule, probability ≥ 0.5 | Evaluation candidates only |
| Context-conditioned SelfCheckGPT-NLI | Fixed threshold 0.5 by default; optional validation F1 tuning | Five independent samples per item |
| Self-consistency batched detector | Fixed majority of five support votes | The same shared answer generation |

The suite has thirteen configured instances: three conventional baselines, eight judges and two shared-answer detectors. The five local/scoring entries are enabled by default. Paid judges are disabled by default and run explicitly one at a time. Each judge has a distinct instance directory and model identifier.

The default generation model is `openai/gpt-4.1-mini`. Five independent completions per exact question/context return five answers, saved in a JSON array with `id`, `question`, and `answer_01` through `answer_05`. Generation excludes the benchmark candidate and its label. Default generation is evaluation only; optional validation generation supports validation-selected SelfCheck thresholds.

SelfCheckGPT uses the official package's recommended `SelfCheckNLI` scorer. Sampling is independent but context-conditioned, so the paper must identify it as a context-conditioned adaptation.

Judges return an integer label and continuous self-reported hallucination probability. Provider-added fields are ignored and recorded; the two required values remain strictly validated. Missing, invalid, truncated, refused, or inconsistent responses stop the run, with raw completions retained for inspection and reuse. There is no fallback label. API client retries default to zero, and only explicit HTTP 429 retries are available. No validation or training API calls occur in the default suite.

Probability baselines report Brier/ECE; normalized similarity and consistency scores do not. Imported binary-only judge predictions omit Brier/ECE and yield only coarse ranking information. AUROC/AUPRC for a binary score are not equivalent to ranking based on continuous confidence. Never choose model settings or thresholds using the frozen evaluation labels.

Model/config hashes, raw input hashes, thresholds, environment information, and aligned predictions are written under `artifacts/baselines/<instance>/`. Reports keep judge models separate. Multi-seed comparisons should use the same fixed data and protocol; not every local baseline supports a suite-wide seed override.
