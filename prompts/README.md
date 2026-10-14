# Prompt files

| Stage | System prompt | User prompt | Input fields |
|---|---|---|---|
| OpenRouter LLM judge (all four judges) | `judge_system.txt` | `judge_user.txt` | question, context, candidate answer |
| Generate the shared five answers | `shared_five_system.txt` | `shared_five_user.txt` | question, context |
| SelfCheckGPT-NLI scoring | No API prompt: local NLI model | Same generated answer file | sampled answer as premise; candidate sentence as hypothesis |
| Self-consistency scoring (not in the paper) | No API prompt: local NLI model | Same generated answer file | majority of whole-answer support votes |

The prompt paths come from `configs/api/*.yaml` and `configs/generation/shared_five.yaml`. The code reads these text files at runtime, then replaces the literal `{{INPUT_JSON}}` in the user prompt with serialized input JSON. Do not remove that marker. Braces inside the input data are safe; Python `str.format` is not used.

## Judge prompt

All four judges (GPT-5 Mini, DeepSeek, Llama 3.3 70B Instruct, Qwen3 32B) use the same system prompt. It states the annotation definition (an answer is hallucinated when at least one material claim is unsupported by, contradicted by, reversed relative to, or fabricated beyond the context; meaning-preserving paraphrase is not hallucination), restricts the model to the supplied context, and requires a JSON object with a binary `hallu_label` and a `probability_hallucinated` value. The decision rule is probability ≥ 0.5. Keep the prompt identical across judges so results stay comparable.

## Shared-answer generation

The shared-generation prompt requests one complete standalone answer. The generator executes it five times independently with distinct request seeds. Both consistency baselines consume the resulting aligned five-answer file; scoring makes no API calls.

## What each stage sees

The generator receives neither the benchmark candidate answer nor its label. The judge receives the benchmark candidate but never its gold label. IDs and questions are added locally to the five-answer JSON export.

## Changing prompts

Prompt contents are included in protocol fingerprints. If you edit them, choose a new output directory before running a new experiment. JSON output contracts are validated in Python even when a provider does not support JSON Schema. Prompt hashes are saved with each run, so keep them with your results.
