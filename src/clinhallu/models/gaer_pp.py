"""Primary GAER++ model and its component-isolated ablation modes."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components.answer_pooling import AnswerRepresentation
from .components.classifier import GAERClassifierHead
from .components.cross_attention import BidirectionalCrossAttention
from .components.grounding import EvidenceStats, GroundingAndEvidencePooling
from .components.interaction import AnswerEvidenceInteraction
from .encoder import MedEncoderV2
from .outputs import GAERPlusPlusOutput

# The ablation ladder is cumulative. B adds answer pooling, C adds
# cross-attention, D adds learned grounding, E exposes evidence statistics,
# and F adds contradiction-sensitive interaction features. Data-pipeline
# ablations belong in configuration rather than in this model.
_GAER_ABLATION_CONFIGS: dict = {
    None: dict(
        use_answer_pool=True,
        use_cross_attn=True,
        use_grounding=True,
        use_contradiction_features=True,
        use_stats=True,
        include_cls=False,
    ),
    "full": dict(
        use_answer_pool=True,
        use_cross_attn=True,
        use_grounding=True,
        use_contradiction_features=True,
        use_stats=True,
        include_cls=False,
    ),
    "F_full_gaer_plus_plus": dict(
        use_answer_pool=True,
        use_cross_attn=True,
        use_grounding=True,
        use_contradiction_features=True,
        use_stats=True,
        include_cls=False,
    ),
    "B_answer_pooling": dict(
        use_answer_pool=True,
        use_cross_attn=False,
        use_grounding=False,
        use_contradiction_features=False,
        use_stats=False,
        include_cls=False,
    ),
    "C_cross_attention": dict(
        use_answer_pool=True,
        use_cross_attn=True,
        use_grounding=False,
        use_contradiction_features=False,
        use_stats=False,
        include_cls=False,
    ),
    "D_grounding": dict(
        use_answer_pool=True,
        use_cross_attn=True,
        use_grounding=True,
        use_contradiction_features=False,
        use_stats=False,
        include_cls=False,
    ),
    "E_evidence_pooling": dict(
        use_answer_pool=True,
        use_cross_attn=True,
        use_grounding=True,
        use_contradiction_features=False,
        use_stats=True,
        include_cls=False,
    ),
    # Experimental only: the primary GAER++ classifier does not include CLS.
    "gaer_plus_plus_with_cls": dict(
        use_answer_pool=True,
        use_cross_attn=True,
        use_grounding=True,
        use_contradiction_features=True,
        use_stats=True,
        include_cls=True,
    ),
}

# Backward-compatible aliases preserve old configs and checkpoints.
_GAER_ABLATION_CONFIGS["deberta_baseline"] = dict(
    use_answer_pool=False,
    use_cross_attn=False,
    use_grounding=False,
    use_contradiction_features=False,
    use_stats=False,
    include_cls=True,
)
_GAER_ABLATION_CONFIGS["answer_pooling"] = _GAER_ABLATION_CONFIGS["B_answer_pooling"]
_GAER_ABLATION_CONFIGS["cross_attention"] = _GAER_ABLATION_CONFIGS["C_cross_attention"]
_GAER_ABLATION_CONFIGS["grounding"] = _GAER_ABLATION_CONFIGS["D_grounding"]
_GAER_ABLATION_CONFIGS["evidence_pooling"] = _GAER_ABLATION_CONFIGS["E_evidence_pooling"]
_GAER_ABLATION_CONFIGS["no_contradiction_features"] = dict(
    use_answer_pool=True,
    use_cross_attn=True,
    use_grounding=True,
    use_contradiction_features=False,
    use_stats=True,
    include_cls=False,
)


class GAERPlusPlusModel(nn.Module):
    """Predict hallucination probability from answer and grounded evidence.

    The primary feature vector is ``[a, e, |a-e|, a*e, cosine, statistics]``.
    It has ``4*hidden_dim + 7`` values (3079 for DeBERTa-v3-base). Enabling
    the optional low-grounding statistic adds one value.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        hidden_dim: int = 768,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attn_implementation: str = "eager",
        lora_config: Optional[dict] = None,
        classifier_hidden_dims: tuple[int, int, int] = (1024, 512, 128),
        classifier_dropouts: tuple[float, float, float] = (0.20, 0.15, 0.10),
        ablation_mode: Optional[str] = None,
        gradient_checkpointing: bool = False,
        grounding_temperature: float = 1.0,
        use_low_grounding_fraction: bool = False,
        answer_pool_attn_hidden_dim: int = 256,
        revision: Optional[str] = None,
    ) -> None:
        super().__init__()
        if ablation_mode not in _GAER_ABLATION_CONFIGS:
            raise ValueError(
                f"Unknown ablation_mode={ablation_mode!r}. "
                f"Valid options: {sorted(k for k in _GAER_ABLATION_CONFIGS if k)}"
            )
        self.ablation_mode = ablation_mode
        self.ablation_flags = _GAER_ABLATION_CONFIGS[ablation_mode]
        self.use_low_grounding_fraction = use_low_grounding_fraction
        self.hidden_dim = hidden_dim

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
        if gradient_checkpointing:
            base = getattr(self.encoder.encoder, "base_model", self.encoder.encoder)
            inner = getattr(base, "model", base)
            if hasattr(inner, "gradient_checkpointing_enable"):
                inner.gradient_checkpointing_enable()

        self.cross_attention = BidirectionalCrossAttention(
            hidden_dim,
            num_attention_heads,
            dropout,
        )
        self.grounding_evidence = GroundingAndEvidencePooling(
            hidden_dim,
            temperature=grounding_temperature,
        )
        self.answer_representation = AnswerRepresentation(
            hidden_dim,
            answer_pool_attn_hidden_dim,
        )
        self.interaction = AnswerEvidenceInteraction()

        interaction_dim = AnswerEvidenceInteraction.output_dim(hidden_dim)  # 4*hidden_dim + 1
        num_stats = EvidenceStats.num_primary_stats(use_low_grounding_fraction)  # 6, or 7

        # Compute the classifier width from its components and guard against
        # accidental feature drift.
        classifier_input_dim = interaction_dim + num_stats
        if self.ablation_flags["include_cls"]:
            classifier_input_dim += hidden_dim
        expected_from_formula = 4 * hidden_dim + 1 + num_stats
        if self.ablation_flags["include_cls"]:
            expected_from_formula += hidden_dim
        assert classifier_input_dim == expected_from_formula, (
            f"Classifier input dim mismatch: computed={classifier_input_dim} "
            f"expected_from_formula={expected_from_formula}"
        )
        if hidden_dim == 768 and not self.ablation_flags["include_cls"]:
            expected_spec_dim = 3080 if use_low_grounding_fraction else 3079
            assert classifier_input_dim == expected_spec_dim, (
                f"For hidden_dim=768 (deberta-v3-base) the primary GAER++ "
                f"feature vector must be {expected_spec_dim}-dim, "
                f"got {classifier_input_dim}."
            )
        self.classifier_input_dim = classifier_input_dim

        self.classifier = GAERClassifierHead(
            input_dim=classifier_input_dim,
            hidden_dims=classifier_hidden_dims,
            dropouts=classifier_dropouts,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_mask: torch.Tensor,
        answer_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> GAERPlusPlusOutput:
        flags = self.ablation_flags
        c, H = self.encoder(input_ids, attention_mask)
        h = H

        if flags["use_cross_attn"]:
            # The reverse branch is not consumed by GAER++, so skip it to save
            # memory and compute on a Colab T4.
            z, _, _, _ = self.cross_attention(
                H,
                context_mask,
                answer_mask,
                attention_mask,
                compute_reverse=False,
                need_weights=False,
            )
        else:
            z = torch.zeros_like(H)

        e, grounding, stats = self.grounding_evidence(
            h,
            z,
            answer_mask,
            attention_mask,
            use_grounding=flags["use_grounding"],
        )
        if not flags["use_cross_attn"]:
            # No context signal at all in this ablation rung -> evidence is zero.
            e = torch.zeros_like(e)

        if flags["use_answer_pool"]:
            # a = Linear([H_CLS ; AttentionPool(H_A)])
            a = self.answer_representation(h, c, answer_mask, attention_mask)
        else:
            a = torch.zeros_like(c)

        interaction_features = self.interaction(
            a,
            e,
            use_contrast=flags["use_contradiction_features"],
        )

        if flags["use_stats"]:
            stats_tensor = stats.stack_primary(self.use_low_grounding_fraction).to(H.dtype)
        else:
            n_stats = 7 if self.use_low_grounding_fraction else 6
            stats_tensor = torch.zeros(H.shape[0], n_stats, device=H.device, dtype=H.dtype)

        feature_parts = [interaction_features, stats_tensor]
        if flags["include_cls"]:
            feature_parts = [c] + feature_parts
        r = torch.cat(feature_parts, dim=-1)
        assert r.shape[-1] == self.classifier_input_dim, (
            f"Runtime feature dim {r.shape[-1]} != expected "
            f"{self.classifier_input_dim} (ablation_mode={self.ablation_mode!r})"
        )

        logits = self.classifier(r)
        probs = torch.sigmoid(logits)

        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype))

        return GAERPlusPlusOutput(
            logits=logits,
            probs=probs,
            p_hallucinated=probs,
            loss=loss,
            c=c,
            a=a,
            e=e,
            grounding=grounding,
            evidence_stats=stats.as_dict(),
            feature_dim=r.shape[-1],
        )
