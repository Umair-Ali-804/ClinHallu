# ClinHallu 2.2.1 GPT-5 Mini hotfix

## Resolved

- Raised the GPT-5 Mini completion limit from 512 to 4096 tokens.
- Set OpenRouter reasoning to `effort: minimal` and excluded returned reasoning.
- Moved GPT-5 Mini to a fresh `openrouter_gpt_5_mini_v3` protocol directory so
  the previous truncated cached response is never reused.
- Added actionable diagnostics containing the provider `finish_reason`, refusal
  state, and reported completion-token count.
- Added offline regression tests for the exact GPT-5 request and truncated-cache
  handling.

No other judge model configuration was changed.

## Verification performed

- `python -m ruff check .`: passed.
- `python -m pytest`: 91 passed, 0 failed.
- Python compilation, all YAML parsing, and the exact GPT-5 Mini request contract:
  passed.
- No paid API request was made.

The single pytest warning is an upstream PyTorch deprecation warning and is not
emitted by ClinHallu code.
