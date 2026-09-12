# Changes in 2.1

- Added the complete `clinhallu pipeline` command while retaining separate prepare, train, evaluate, condition, ablation, baseline and report commands.
- Added OpenRouter judges with eight explicit model configurations, external prompt files, strict required-field validation, raw-response caches, evaluation-only API requests and continuous score exports.
- Updated `clinhallu generate` to create five independent stochastic completions per question/context, with resumable request caches, an optional per-run request limit, protocol fingerprints, and an aligned five-answer export. Evaluation-only generation is the default; validation is optional.
- Replaced the custom SelfCheck proxy with the official package's recommended `SelfCheckNLI` interface. The distinct majority-vote self-consistency detector remains a separately named adaptation.
- Wired thirteen baseline instances into the suite: three conventional baselines, eight requested OpenRouter judges, SelfCheckGPT-NLI, and self-consistency. Paid judges are disabled by default and remain selectable one at a time with `--only`.
- Fixed the missing `torch.nn` import in interaction features, FP16 grounding-statistic underflow, and frozen-evaluation temperature scaling from raw logits. Final prediction writing precedes the evaluation completion marker.
- Preserved numeric IDs in baseline loading and generation exports; tightened JSON record type checks.
- Separated judge model names in baseline report aggregation. Removed probability calibration metrics from imported label-only judgments.
- Updated dependency compatibility, README commands, Colab notebook, prompt documentation and scientific protocol notes. Checkpoint parameter names remain unchanged; numerical precision fixes can affect outputs, so record this code version in new experiments.

The test suite includes local actual training/evaluation for the full detector and every B–F ablation, local NLI/embedding runtime checks, and mocked API integration. See `validation_report.md`. Full research runs require the user's fixed datasets and, for API stages, a funded OpenRouter key.
