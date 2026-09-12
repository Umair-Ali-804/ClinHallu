# Validation report

Date: 2026-09-10. This report concerns the code in this package, not research-model performance.

## Executed checks

- The restored unit, integration, protocol, and regression tests pass after the 2.2.1 fixes.
- API/protocol checks cover five independent generation requests, cache reuse, duplicate question/context sharing, integer ID preservation, interrupted-run recovery, generation limits, validation opt-in, malformed/truncated-response caching and diagnostics, prompt/data/output drift detection, strict required judge fields with harmless extra-field tolerance, continuous score metrics, separate judge report rows, all thirteen suite configurations, exact requested model IDs, GPT-5 parameter compatibility and reasoning budget, and finite FP16 grounding gradients.
- Eight added runtime tests passed using real locally created tiny model weights. These execute training, validation calibration and frozen evaluation for GAER++ and all five B–F ablations; checkpoint/safetensors export; completed-run reuse; a smoke run that excludes frozen evaluation; TF-IDF fitting and report creation; real local NLI inference; and real local sentence-embedding/NLI baseline evaluation.

**Final combined run: 91 passed, 0 failed, 1 upstream dependency warning, in 92.07 seconds.** The runtime tests use a one-layer, 16-dimensional DeBERTa model, a locally created tokenizer, balanced synthetic splits (8 train, 4 validation, 4 evaluation), and one training epoch. They run the actual trainer/calibration/evaluation subprocesses. Synthetic scores must not be reported as ClinHallu benchmark performance.

## Environment

Python 3.12; CPU execution (`torch.cuda.is_available() == False`). Main tested packages: PyTorch 2.14.0, Transformers 4.57.6, PEFT 0.18.1, Accelerate 1.15.0, tokenizers 0.22.2, huggingface-hub 0.36.2, safetensors 0.8.0, sentence-transformers 5.7.0, SelfCheckGPT 0.1.7, NumPy 2.3.5, scikit-learn 1.8.0, pytest 9.1.1. Colab requirements retain NumPy 2.2.6/pandas 2.2.3 and the existing Colab PyTorch/CUDA installation; that exact GPU stack was not exercised here.

The remaining warning is emitted inside an upstream dependency for deprecated `torch.jit.script`; it is not a failed assertion or a call in ClinHallu source code.

## Not executed

The original full research datasets were not included. No full pretrained DeBERTa C1–C5 experiment was trained, no production GPU run was executed, and no paid OpenRouter completion was made. API integration was checked with mocked responses and mocked HTTP transport. All eight model IDs were also found in OpenRouter's public catalog on 2026-09-10. Live account access, credits, provider availability and model quality must still be verified in the user's environment.

## Protocol limits

The five answers are independently sampled but context-conditioned. SelfCheckGPT therefore remains a context-conditioned adaptation even though it uses the official NLI scorer. It uses a fixed threshold by default; validation tuning requires explicitly generated validation answers. Self-consistency always uses fixed majority voting. Neither method chooses a threshold using test labels. Cached responses must remain with their manifests.

## Reproduce

From the project directory after installing the development dependencies:

```bash
python -m pytest
```

No network or real API key is required by the added tests. Local tiny models are created under pytest's temporary directories; their artifacts are excluded from the deliverable.
