# OpenRouter and method references

This package uses OpenRouter's chat-completions HTTP endpoint directly; no API SDK is required. Model availability should be checked immediately before a paid run.

All eight configured model pages were available in OpenRouter's public catalog on 2026-09-10. Availability and routing can still change, so use `clinhallu api models --contains <substring>` immediately before a paid run.

- [OpenRouter API reference](https://openrouter.ai/docs/api_reference/overview): Bearer authentication, `/api/v1/chat/completions`, messages, model IDs, response content and usage.
- [Structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs): `response_format.json_schema`, strict schemas, and `provider.require_parameters`. Support is endpoint-dependent; local schema validation remains necessary.
- [Reasoning controls](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens): optional reasoning configuration, per-model capabilities, and the distinction between disabling reasoning and merely hiding it. GPT-5 Mini uses `effort: minimal` with a larger completion allowance because reasoning tokens count against `max_tokens`; Gemini/Qwen disable optional reasoning for their short JSON judgment budget.
- GPT-5 Mini: `openai/gpt-5-mini`.
- Gemini 2.5 Flash: `google/gemini-2.5-flash`.
- DeepSeek Chat: `deepseek/deepseek-chat`.
- Llama 3.3 70B Instruct: `meta-llama/llama-3.3-70b-instruct`.
- Qwen3 32B: `qwen/qwen3-32b`.
- Llama 4 Maverick: `meta-llama/llama-4-maverick`.
- Qwen3 30B A3B: `qwen/qwen3-30b-a3b`.
- Gemini 2.5 Flash Lite: `google/gemini-2.5-flash-lite`.
- SelfCheckGPT sampling generator: `openai/gpt-4.1-mini`.
- [SelfCheckGPT paper](https://arxiv.org/abs/2303.08896): sampling-based hallucination detection.
- [Self-consistency paper](https://arxiv.org/abs/2203.11171): aggregation of sampled reasoning paths.

Run `clinhallu api models --contains <substring>` to inspect current model IDs, supported parameters, reasoning metadata, context length and listed pricing. Hosted model IDs do not provide the same immutable-weight guarantee as a pinned local checkpoint. Preserve the actual returned model/provider/completion metadata and prompt hashes for each experiment. No model quality or price advantage is asserted by choosing these configurable examples.
