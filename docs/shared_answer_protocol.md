# Five-sample SelfCheckGPT protocol

## Independent generation

The protocol identifier is `selfcheckgpt_five_independent_samples_v1`. For each exact question/context pair, the project issues five separate OpenRouter completions to `openai/gpt-4.1-mini`, using temperature 1.0 and seeds 13–17. Each request returns one `answer` string. The candidate answer, ID, and gold label are excluded from the API input.

The five results are exported as `answer_01` through `answer_05`. The manifest records the generator, prompts, settings, request hashes, source hash, and output hash. Evaluation generation is the default; validation generation is opt-in.

## SelfCheckGPT-NLI

The candidate `answer` is split into sentences. The implementation then calls the official package's recommended interface:

```python
selfcheck_nli.predict(
    sentences=candidate_sentences,
    sampled_passages=five_independent_answers,
)
```

This returns one non-factuality score per candidate sentence. The document score is the arithmetic mean of those sentence scores. Higher values indicate greater hallucination risk. The fixed default decision rule is `score >= 0.5`; alternatively, a threshold may be selected using the separate validation split. Evaluation labels must never select the threshold.

Because the sampled passages are conditioned on the supplied evidence context, report the method as a **context-conditioned SelfCheckGPT-NLI adaptation**. If the original candidate and sampled passages come from different generators, also disclose the cross-model setting.

## Self-consistency adaptation

The separate self-consistency detector may reuse the same five stored answers, but it retains its custom NLI voting implementation. It must not be described as part of the official SelfCheckGPT implementation.

## Integrity and cost

For 1,000 unique evaluation question/context pairs, a fresh SelfCheckGPT generation cache requires 5,000 successful OpenRouter completions. Every completion is cached immediately, so an interrupted run can resume. Local SelfCheckGPT scoring adds no OpenRouter calls.

SelfCheckGPT scores are normalized inconsistency scores rather than calibrated hallucination probabilities, so Brier score and ECE are omitted. Each run saves aligned predictions, metrics, source and sample hashes, sentence-level scores, and reproducibility metadata.
