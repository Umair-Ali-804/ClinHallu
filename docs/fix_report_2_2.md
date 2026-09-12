# ClinHallu 2.2 repair report

## Resolved

- Integrated the OpenRouter GPT-5 Mini temperature fix into the complete project.
- Integrated tolerant handling of provider-added judge JSON fields without weakening required-value checks.
- Verified exactly eight requested judge IDs and removed obsolete judge models/configurations.
- Protected one-at-a-time paid execution by disabling judge entries in default bulk runs.
- Preserved per-model API caches and resume behavior.
- Verified official SelfCheckGPT 0.1.7 import with Transformers 4.57.6.
- Corrected stale SelfCheckGPT/self-consistency cache assumptions.
- Aligned package, requirements, environment checks, CLI help, notebooks, and documentation.
- Added aligned-length checks to prevent silent prediction/result truncation.
- Corrected report-table closure behavior and the deprecated SentenceTransformers import.
- Formatted the codebase and enabled correctness-focused lint rules.

## Verification performed

- `python -m ruff check .`: passed.
- `python -m pytest`: 90 passed, 0 failed.
- Editable package installation: passed.
- `python -m pip check`: passed.
- Environment inspection: passed on CPU.
- SelfCheckGPT import: passed.
- All YAML files and notebooks parsed successfully.
- All eight judge configs completed offline request-construction dry runs.
- All five GAER++ ablation paths executed with tiny local weights in integration tests.
- Conventional baseline, report, frozen-evaluation, cache/resume, and data-protocol tests passed.
- No paid API requests or research benchmark runs were performed.

The only test warning comes from deprecated `torch.jit.script` usage inside an upstream dependency. It is not emitted by ClinHallu source code and does not affect the test result.
