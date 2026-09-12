"""DeBERTa encoder loading, LoRA attachment, and trainability audits."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

logger = logging.getLogger(__name__)

_DEFAULT_LORA_TARGET_MODULES = [
    "query_proj",
    "key_proj",
    "value_proj",
    "attention.output.dense",
]


def verify_lora_target_modules(base_model: nn.Module, target_modules: list) -> None:
    """Verify configured LoRA target modules exist in the loaded encoder."""
    all_names = [n for n, _ in base_model.named_modules()]
    missing = [t for t in target_modules if not any(n.endswith(t) for n in all_names)]
    if not missing:
        return

    sample_names = [n for n in all_names if "." in n][:30]
    raise ValueError(
        f"LoRA target_modules {missing} do not exist as a suffix of any module name in the "
        f"loaded encoder. A sample of actual module names on this model: {sample_names}. Run "
        f'`python -c "from transformers import AutoModel; '
        f"m = AutoModel.from_pretrained('<encoder>'); "
        f'print([n for n, _ in m.named_modules()])"` to see the full list for your installed '
        f"Transformers version before configuring LoRA."
    )


def verify_trainable_parameters(model: nn.Module) -> dict:
    """Audit trainable and frozen encoder, LoRA, and head parameters."""
    base_trainable = 0
    base_frozen = 0
    lora_trainable = 0
    other_trainable = 0

    for name, p in model.named_parameters():
        is_lora = "lora_" in name
        is_encoder = name.startswith("encoder.")

        if is_encoder and is_lora:
            if p.requires_grad:
                lora_trainable += p.numel()
        elif is_encoder:
            if p.requires_grad:
                base_trainable += p.numel()
            else:
                base_frozen += p.numel()
        elif p.requires_grad:
            other_trainable += p.numel()

    logger.info(
        "Trainable-parameter audit: base_encoder_trainable=%d (should be 0 when using LoRA) "
        "base_encoder_frozen=%d lora_trainable=%d gaer_modules_trainable=%d",
        base_trainable,
        base_frozen,
        lora_trainable,
        other_trainable,
    )

    return {
        "base_encoder_trainable_params": base_trainable,
        "base_encoder_frozen_params": base_frozen,
        "lora_trainable_params": lora_trainable,
        "gaer_module_trainable_params": other_trainable,
    }


class MedEncoderV2(nn.Module):
    """Transformer encoder returning the CLS vector and the full token sequence."""

    def __init__(
        self,
        model_name: str,
        dropout: float = 0.1,
        attn_implementation: str = "eager",
        lora_config: dict | None = None,
        revision: str | None = None,
    ) -> None:
        super().__init__()

        revision_kwargs = {"revision": revision} if revision else {}
        config = AutoConfig.from_pretrained(
            model_name,
            output_attentions=False,
            attn_implementation=attn_implementation,
            **revision_kwargs,
        )
        base_encoder = AutoModel.from_pretrained(model_name, config=config, **revision_kwargs)

        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.uses_lora = False

        if lora_config is None:
            self.encoder = base_encoder
        else:
            self.encoder = self._attach_lora(base_encoder, lora_config)
            self.uses_lora = True

        logger.info(
            "MedEncoderV2 loaded: %s hidden=%d lora=%s",
            model_name,
            self.hidden_size,
            self.uses_lora,
        )

    def _attach_lora(self, base_encoder: nn.Module, lora_config: dict) -> nn.Module:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "lora_config was provided but `peft` is not installed. "
                "Run `pip install peft` or pass lora_config=None."
            ) from exc

        target_modules = lora_config.get("target_modules", _DEFAULT_LORA_TARGET_MODULES)
        verify_lora_target_modules(base_encoder, target_modules)

        peft_cfg = LoraConfig(
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("lora_alpha", 32),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            bias=lora_config.get("bias", "none"),
            target_modules=target_modules,
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        encoder = get_peft_model(base_encoder, peft_cfg)

        # Without this, LoRA parameters inside gradient-checkpointed layers receive no
        # gradients, because the checkpointed block sees no input with requires_grad=True.
        if hasattr(encoder, "enable_input_require_grads"):
            encoder.enable_input_require_grads()
            logger.info("MedEncoderV2: enabled input gradients for LoRA + gradient checkpointing")
        else:
            logger.warning(
                "MedEncoderV2: encoder does not expose enable_input_require_grads(); "
                "gradient checkpointing may not propagate gradients to LoRA parameters."
            )

        trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        total = sum(p.numel() for p in encoder.parameters())
        logger.info(
            "MedEncoderV2: LoRA applied - trainable %d / %d params (%.3f%%)",
            trainable,
            total,
            100.0 * trainable / total,
        )
        return encoder

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        cls_vector = self.dropout(hidden_states[:, 0, :])
        return cls_vector, hidden_states

    def freeze_base(self) -> None:
        """Freeze all encoder parameters."""
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_base(self) -> None:
        """Unfreeze all encoder parameters."""
        for p in self.encoder.parameters():
            p.requires_grad = True
