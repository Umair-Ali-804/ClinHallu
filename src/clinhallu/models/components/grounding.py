"""Learned token grounding and context-derived evidence pooling."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class EvidenceStats:
    """Per-example grounding scores and pooling-distribution statistics."""

    mean: torch.Tensor = field(default=None)
    max: torch.Tensor = field(default=None)
    min: torch.Tensor = field(default=None)
    std: torch.Tensor = field(default=None)
    entropy: torch.Tensor = field(default=None)
    coverage: torch.Tensor = field(default=None)
    concentration: torch.Tensor = field(default=None)
    high_grounding_fraction: torch.Tensor = field(default=None)
    low_grounding_fraction: torch.Tensor = field(default=None)

    # Fixed order for statistics that enter the primary classifier.
    PRIMARY_STAT_NAMES = ["mean", "max", "min", "std", "entropy", "coverage"]

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "grounding_mean": self.mean,
            "grounding_max": self.max,
            "grounding_min": self.min,
            "grounding_std": self.std,
            "grounding_entropy": self.entropy,
            "grounding_coverage": self.coverage,
            "grounding_concentration": self.concentration,
            "grounding_high_fraction": self.high_grounding_fraction,
            "grounding_low_fraction": self.low_grounding_fraction,
        }

    def stack_primary(
        self,
        include_low_grounding_fraction: bool = False,
    ) -> torch.Tensor:
        """Stack classifier statistics in their stable feature order."""
        tensors = [getattr(self, name) for name in self.PRIMARY_STAT_NAMES]
        if include_low_grounding_fraction:
            tensors.append(self.low_grounding_fraction)
        return torch.stack(tensors, dim=-1)

    @classmethod
    def num_primary_stats(cls, include_low_grounding_fraction: bool = False) -> int:
        return len(cls.PRIMARY_STAT_NAMES) + (1 if include_low_grounding_fraction else 0)


class GroundingAndEvidencePooling(nn.Module):
    """Pool context-derived evidence with learned answer-token grounding.

    For each answer token i:
        ``u_i = [h_i; z_i; |h_i-z_i|; h_i*z_i]``
        ``g_logit_i = W_g u_i + b_g``
        ``alpha_i = softmax(g_logit_i / temperature)``
        ``e = sum(alpha_i * z_i)``

    Sigmoid scores are retained for interpretation, while softmax logits set
    the pooling weights. Disabling grounding uses uniform answer-token weights.
    The pooled values always come from ``z`` (context), never from ``h``.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        temperature: float = 1.0,
        low_grounding_threshold: float = 0.35,
        high_grounding_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.low_grounding_threshold = low_grounding_threshold
        self.high_grounding_threshold = high_grounding_threshold
        self.grounding_proj = nn.Linear(4 * hidden_dim, 1)

    def compute_grounding_logit(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        u = torch.cat([h, z, (h - z).abs(), h * z], dim=-1)
        return self.grounding_proj(u).squeeze(-1)

    def forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        answer_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        use_grounding: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, EvidenceStats]:
        real = attention_mask.bool()
        amask = answer_mask.bool() & real
        amask_f = amask.float()
        n_answer = amask_f.sum(dim=1).clamp(min=1.0)
        has_answer = amask.any(dim=1)

        g_logit = self.compute_grounding_logit(h, z).float()
        g = torch.sigmoid(g_logit)

        if use_grounding:
            # A finite fill prevents NaNs in a fully masked row; that row is
            # explicitly zeroed after pooling.
            neg_fill = torch.finfo(g_logit.dtype).min
            masked_logits = torch.where(
                amask,
                g_logit / self.temperature,
                torch.full_like(g_logit, neg_fill),
            )
            alpha = torch.softmax(masked_logits, dim=1)
            alpha = alpha * amask_f
        else:
            alpha = amask_f / n_answer.unsqueeze(1)

        # FP32 prevents epsilon underflow and infinite gradients in half precision.
        with torch.autocast(device_type=h.device.type, enabled=False):
            e = torch.einsum("bn,bnd->bd", alpha, z.float()).to(h.dtype)
        e = torch.where(has_answer.unsqueeze(-1), e, torch.zeros_like(e))

        stats = self._evidence_stats(
            g,
            alpha,
            amask,
            amask_f,
            n_answer,
            has_answer,
            self.low_grounding_threshold,
            self.high_grounding_threshold,
        )
        return e, g, stats

    @staticmethod
    def _evidence_stats(
        g: torch.Tensor,
        alpha: torch.Tensor,
        amask: torch.Tensor,
        amask_f: torch.Tensor,
        n_answer: torch.Tensor,
        has_answer: torch.Tensor,
        low_threshold: float,
        high_threshold: float,
    ) -> EvidenceStats:
        g_masked = g * amask_f
        mean_g = g_masked.sum(dim=1) / n_answer

        dtype_min = torch.finfo(g.dtype).min
        dtype_max = torch.finfo(g.dtype).max
        g_for_max = torch.where(amask, g, torch.full_like(g, dtype_min))
        max_g = g_for_max.max(dim=1).values
        g_for_min = torch.where(amask, g, torch.full_like(g, dtype_max))
        min_g = g_for_min.min(dim=1).values

        var = ((g_masked - mean_g.unsqueeze(1)) ** 2 * amask_f).sum(dim=1) / n_answer
        # sqrt'(0) is infinite -- add a small epsilon rather than clamping to
        # a min of exactly 0 (regression-tested: this previously produced
        # NaN gradients on any single-answer-token example).
        std_g = (var.clamp(min=0.0) + 1e-12).sqrt()

        # Entropy and concentration describe normalized pooling weights.
        alpha_safe = alpha.clamp(min=1e-8)
        entropy = -(alpha_safe * torch.log(alpha_safe) * amask_f).sum(dim=1)
        alpha_masked = torch.where(amask, alpha, torch.zeros_like(alpha))
        concentration = alpha_masked.max(dim=1).values

        # Coverage is mean grounding; the thresholded version stays separate.
        coverage = mean_g
        high_frac = ((g > high_threshold).to(g.dtype) * amask_f).sum(dim=1) / n_answer
        low_frac = ((g < low_threshold).to(g.dtype) * amask_f).sum(dim=1) / n_answer

        zero = torch.zeros_like(mean_g)
        return EvidenceStats(
            mean=torch.where(has_answer, mean_g, zero),
            max=torch.where(has_answer, max_g, zero),
            min=torch.where(has_answer, min_g, zero),
            std=torch.where(has_answer, std_g, zero),
            entropy=torch.where(has_answer, entropy, zero),
            coverage=torch.where(has_answer, coverage, zero),
            concentration=torch.where(has_answer, concentration, zero),
            high_grounding_fraction=torch.where(has_answer, high_frac, zero),
            low_grounding_fraction=torch.where(has_answer, low_frac, zero),
        )
