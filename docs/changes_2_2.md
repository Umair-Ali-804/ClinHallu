# Changes in 2.2

- Replaced the obsolete judge list with the eight requested OpenRouter model IDs.
- Added a GPT-5-specific request configuration that omits unsupported custom temperature.
- Relaxed judge object shape validation only for harmless provider-added fields; required label and probability values remain strict, finite, bounded, and mutually consistent.
- Preserved immediate raw-response caching and resume behavior, so accepted cached completions are reused without another paid request.
- Disabled all paid judge entries by default while retaining explicit one-at-a-time `--only` execution.
- Kept DeepSeek in a dedicated output/cache directory and kept every judge result separate.
- Added strict length checking to aligned generation, prediction, consistency, evaluation, and report writes.
- Corrected the official SelfCheckGPT versus self-consistency protocol tests: both reuse the same five sampled answers but use distinct local NLI scorers.
- Added SelfCheckGPT to package metadata and environment validation.
- Standardized the supported Colab stack on Transformers 4.57.6 and PEFT 0.18.1.
- Updated stale model names, suite counts, CLI help, Colab instructions, and protocol documentation.
- Applied consistent formatting and enabled high-signal Ruff correctness checks.

Validation used real tiny local neural models, all five ablations, conventional baselines, mocked OpenRouter transport, cache/resume checks, strict split tests, packaging checks, and all command/config dry runs. No paid API request was made.
