"""Construct a configured model in one place for training and evaluation."""

from __future__ import annotations

from typing import Mapping

from .gaer_pp import GAERPlusPlusModel
from .legacy_hkg import HKGFusionModelV2


def build_model(cfg: Mapping, *, gradient_checkpointing: bool | None = None):
    model_cfg = cfg["model"]
    common = {
        "model_name": model_cfg["encoder"],
        "hidden_dim": int(model_cfg["hidden_dim"]),
        "dropout": float(model_cfg["dropout"]),
        "attn_implementation": model_cfg.get("attn_implementation", "eager"),
        "lora_config": cfg.get("lora"),
        "revision": model_cfg.get("revision"),
    }
    if model_cfg["type"] == "gaer_plus_plus":
        return GAERPlusPlusModel(
            **common,
            num_attention_heads=int(model_cfg["num_attention_heads"]),
            ablation_mode=model_cfg.get("ablation_mode"),
            gradient_checkpointing=(
                bool(cfg["training"].get("gradient_checkpointing", False))
                if gradient_checkpointing is None
                else gradient_checkpointing
            ),
            grounding_temperature=float(cfg.get("grounding", {}).get("temperature", 1.0)),
            use_low_grounding_fraction=bool(
                cfg.get("grounding", {}).get("use_low_grounding_fraction", False)
            ),
        )
    return HKGFusionModelV2(
        **common,
        classifier_hidden_dim=int(model_cfg.get("classifier_hidden_dim", 512)),
        num_attention_heads=int(model_cfg["num_attention_heads"]),
        ablation_mode=model_cfg.get("ablation_mode"),
    )
