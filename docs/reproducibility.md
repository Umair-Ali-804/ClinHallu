# Reproducibility

Every model run writes:

- `resolved_config.yaml` with all inherited settings expanded;
- `run_manifest.json` with raw data hashes and split-use restrictions;
- `environment.json` with Python, CUDA, GPU and library versions;
- `run_state.json` with the completed lifecycle stages;
- training history, checkpoints, calibration files, metrics and predictions.

For Colab, install `requirements/colab.txt` first and then install the package
with `pip install -e . --no-deps`. This preserves Colab's working PyTorch/CUDA
installation.

The primary DeBERTa encoder and the baseline models use immutable Hugging Face
commit hashes from the supplied study configuration. Change a revision only as
a new, explicitly named experiment.

Before publishing, archive the exact project code, resolved configuration,
`model_only.safetensors`, calibration files, metrics, predictions and manifest.

The OpenRouter judges and generator use explicit hosted model IDs. Hosted weights/provider routing can change; their model IDs are not immutable commit hashes. Preserve raw completion caches and returned model/provider metadata together with the generation and scoring manifests. SelfCheckGPT-NLI and self-consistency reuse the same five generated answers but use distinct pinned local NLI scoring implementations.

Resume restores epoch-boundary training state but not all RNG and DataLoader state. It supports continued training, not a promise of bit-for-bit identity with an uninterrupted stochastic run.
