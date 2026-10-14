# OpenRouter and method references

This package uses OpenRouter's chat-completions HTTP endpoint directly; no API SDK is required. Model availability should be checked immediately before a paid run.

All four configured judge model pages were available in OpenRouter's public catalog on 2026-09-10. Availability and routing can still change, so use `clinhallu api models --contains <substring>` immediately before a paid run.

- [OpenRouter API reference](https://openrouter.ai/docs/api_reference/overview): Bearer authentication, `/api/v1/chat/completions`, messages, model IDs, response content and usage.
- [Structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs): `response_format.json_schema`, strict schemas, and `provider.require_parameters`. Support is endpoint-dependent; local schema validation remains necessary.
- [Reasoning controls](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens): optional reasoning configuration, per-model capabilities, and the distinction between disabling reasoning and merely hiding it. GPT-5 Mini uses `effort: minimal` with a larger completion allowance because reasoning tokens count against `max_tokens`; Qwen3 disables optional reasoning for its short JSON judgment budget.

**Zero-shot LLM judges (as used in the paper)**

- GPT-5 Mini: `openai/gpt-5-mini`.
- DeepSeek V3.2: `deepseek/deepseek-chat` (the paper reports DeepSeek-V3.2; confirm what this alias resolves to before a run).
- Llama 3.3 70B Instruct: `meta-llama/llama-3.3-70b-instruct`.
- Qwen3 32B: `qwen/qwen3-32b`.

**Sampling generator**

- SelfCheckGPT sampling generator: `openai/gpt-4.1-mini` (the package default; configurable in `configs/generation/shared_five.yaml`).

**Method references**

- [SelfCheckGPT paper](https://arxiv.org/abs/2303.08896): sampling-based hallucination detection.
- [Self-consistency paper](https://arxiv.org/abs/2203.11171): aggregation of sampled reasoning paths (used by the optional self-consistency baseline, which is not part of the paper's results).

Run `clinhallu api models --contains <substring>` to inspect current model IDs, supported parameters, reasoning metadata, context length and listed pricing. Hosted model IDs do not provide the same immutable-weight guarantee as a pinned local checkpoint. Preserve the actual returned model/provider/completion metadata and prompt hashes for each experiment, since requests may be routed across several serving backends within one run. No model quality or price advantage is asserted by choosing these configurable examples.
