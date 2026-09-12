"""Dependency-free validation for experiment configuration dictionaries."""

from __future__ import annotations

from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised when a run configuration is incomplete or inconsistent."""


def _require(mapping: Mapping[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing configuration value: {section}.{key}")
    return mapping[key]


def validate_config(cfg: Mapping[str, Any]) -> None:
    for section in ("experiment", "model", "data", "training", "evaluation", "output"):
        value = _require(cfg, section, "root")
        if not isinstance(value, Mapping):
            raise ConfigError(f"Configuration section '{section}' must be a mapping")

    experiment = cfg["experiment"]
    _require(experiment, "name", "experiment")
    kind = _require(experiment, "kind", "experiment")
    if kind not in {"adaptation", "ablations"}:
        raise ConfigError("experiment.kind must be 'adaptation' or 'ablations'")

    model = cfg["model"]
    if _require(model, "type", "model") not in {"gaer_plus_plus", "hkg_fusion_v2"}:
        raise ConfigError("model.type must be gaer_plus_plus or hkg_fusion_v2")
    if int(_require(model, "max_length", "model")) < 16:
        raise ConfigError("model.max_length must be at least 16")
    hidden_dim = int(_require(model, "hidden_dim", "model"))
    attention_heads = int(_require(model, "num_attention_heads", "model"))
    if hidden_dim < 1 or attention_heads < 1 or hidden_dim % attention_heads:
        raise ConfigError("model.hidden_dim must be positive and divisible by num_attention_heads")

    data = cfg["data"]
    for key in ("train_path", "val_path", "eval_path", "processed_dir", "cache_dir"):
        _require(data, key, "data")
    if int(data.get("question_budget", 0)) < 1 or int(data.get("answer_budget", 0)) < 1:
        raise ConfigError("question_budget and answer_budget must be positive")
    if int(data["question_budget"]) + int(data["answer_budget"]) + 4 > int(model["max_length"]):
        raise ConfigError(
            "question_budget + answer_budget + four special tokens exceeds max_length"
        )
    expected_rows = data.get("expected_rows", {})
    if set(expected_rows) - {"train", "validation", "test"}:
        raise ConfigError("expected_rows keys must be train, validation, and test")
    if any(int(value) < 1 for value in expected_rows.values() if value is not None):
        raise ConfigError("expected row counts must be positive")

    seed = int(_require(cfg["training"], "seed", "training"))
    if seed < 0:
        raise ConfigError("training.seed must be non-negative")
    if int(cfg["training"].get("gradient_accumulation_steps", 1)) < 1:
        raise ConfigError("gradient_accumulation_steps must be at least 1")
    for key in ("batch_size", "validation_batch_size", "max_epochs"):
        if int(_require(cfg["training"], key, "training")) < 1:
            raise ConfigError(f"training.{key} must be positive")
    if float(cfg.get("grounding", {}).get("temperature", 1.0)) <= 0:
        raise ConfigError("grounding.temperature must be positive")
    if int(_require(cfg["evaluation"], "batch_size", "evaluation")) < 1:
        raise ConfigError("evaluation.batch_size must be positive")
    labels = cfg.get("labels", {})
    if labels.get("grounded") != 0 or labels.get("hallucinated") != 1:
        raise ConfigError("Label semantics are fixed: grounded=0, hallucinated=1")
