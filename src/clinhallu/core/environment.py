"""Capture the environment used for a reproducible run."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from pathlib import Path

PACKAGES = (
    "accelerate",
    "huggingface-hub",
    "joblib",
    "matplotlib",
    "numpy",
    "pandas",
    "peft",
    "PyYAML",
    "safetensors",
    "scikit-learn",
    "scipy",
    "seaborn",
    "sentence-transformers",
    "sentencepiece",
    "selfcheckgpt",
    "statsmodels",
    "tokenizers",
    "torch",
    "tqdm",
    "transformers",
)


def environment_record() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    torch_error = None
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda
        gpu = torch.cuda.get_device_name(0) if cuda_available else None
    except Exception as exc:
        cuda_available, cuda_version, gpu = False, None, None
        torch_error = f"{type(exc).__name__}: {exc}"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "gpu": gpu,
        "torch_error": torch_error,
    }


def write_environment(path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(environment_record(), indent=2), encoding="utf-8")
