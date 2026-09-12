"""Legacy HKG-Fusion model retained for checkpoint compatibility."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components.classifier import FeatureConcatenation, HallucinationClassifier
from .components.context_attention import ContextAwareAttention
from .components.evidence_pooling import EvidenceAwarePooling
from .encoder import MedEncoderV2
from .outputs import HKGFusionOutput


class HKGFusionModelV2(nn.Module):
    """Original two-class HKG-Fusion architecture.

    Attribute names and tensor operations are unchanged from the supplied
    project so old ``best_model.pt`` state dictionaries remain compatible.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        hidden_dim: int = 768,
        classifier_hidden_dim: int = 512,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attn_implementation: str = "eager",
        lora_config: Optional[dict] = None,
        bce_eps: float = 1e-7,
        ablation_mode: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> None:
        super().__init__()
        if ablation_mode not in (None, "no_evidence"):
            raise ValueError(f"Unknown legacy ablation_mode={ablation_mode!r}")
        self.ablation_mode = ablation_mode
        self.encoder = MedEncoderV2(
            model_name,
            dropout,
            attn_implementation,
            lora_config,
            revision=revision,
        )
        if self.encoder.hidden_size != hidden_dim:
            raise ValueError(
                f"Configured hidden_dim={hidden_dim}, but {model_name} exposes "
                f"hidden_size={self.encoder.hidden_size}."
            )
        self.context_attention = ContextAwareAttention(hidden_dim, num_attention_heads, dropout)
        self.evidence_pooling = EvidenceAwarePooling(hidden_dim)
        self.concat = FeatureConcatenation()
        self.classifier = HallucinationClassifier(
            input_dim=2 * hidden_dim,
            hidden_dim=classifier_hidden_dim,
            num_classes=2,
            dropout=dropout,
        )
        self.bce_eps = bce_eps

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_mask: torch.Tensor,
        answer_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> HKGFusionOutput:
        cls, hidden = self.encoder(input_ids, attention_mask)
        attended, support_score = self.context_attention(
            hidden, context_mask, answer_mask, attention_mask
        )
        evidence = self.evidence_pooling(attended, answer_mask, support_score)
        if self.ablation_mode == "no_evidence":
            evidence = torch.zeros_like(evidence)

        logits = self.classifier(self.concat(cls, evidence))
        probs = F.softmax(logits, dim=-1)
        p_hallucinated = probs[:, 1]
        loss = None
        if labels is not None:
            targets = labels.to(p_hallucinated.dtype)
            values = p_hallucinated.clamp(self.bce_eps, 1.0 - self.bce_eps)
            loss = -(targets * torch.log(values) + (1.0 - targets) * torch.log(1.0 - values)).mean()

        return HKGFusionOutput(
            logits=logits,
            probs=probs,
            p_hallucinated=p_hallucinated,
            loss=loss,
            c=cls,
            e=evidence,
        )
