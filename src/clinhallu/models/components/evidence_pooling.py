"""Legacy support-weighted evidence pooling for HKG-Fusion checkpoints."""

from __future__ import annotations

import torch
import torch.nn as nn


class EvidenceAwarePooling(nn.Module):
    def __init__(self, hidden_dim: int = 768, support_floor: float = 0.05) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.support_floor = support_floor

    def forward(
        self,
        attended_H: torch.Tensor,
        answer_mask: torch.Tensor,
        support_score: torch.Tensor,
    ) -> torch.Tensor:
        answer_mask = answer_mask.bool()
        gate = support_score.clamp(min=0.0, max=1.0)
        gate = self.support_floor + (1.0 - self.support_floor) * gate
        gated_H = attended_H * gate.unsqueeze(-1)

        neg_inf = torch.finfo(gated_H.dtype).min
        mask3d = answer_mask.unsqueeze(-1)
        masked_H = torch.where(mask3d, gated_H, torch.full_like(gated_H, neg_inf))

        e, _ = masked_H.max(dim=1)

        has_answer = answer_mask.any(dim=1, keepdim=True)
        e = torch.where(has_answer, e, torch.zeros_like(e))

        return e
