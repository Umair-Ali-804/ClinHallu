"""Answer/evidence interaction features used by the GAER++ classifier."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnswerEvidenceInteraction(nn.Module):
    """Build ``[a; e; |a-e|; a*e; cosine(a,e)]``.

    Output dim = 4*hidden_dim + 1 (the final +1 is the scalar cosine similarity).
    """

    def forward(self, a: torch.Tensor, e: torch.Tensor, use_contrast: bool = True) -> torch.Tensor:
        if not use_contrast:
            # Preserve classifier width so checkpoints share one shape across
            # ablation modes.
            zeros_pair = torch.zeros_like(a)
            zero_cos = torch.zeros(a.shape[0], 1, device=a.device, dtype=a.dtype)
            return torch.cat([a, e, zeros_pair, zeros_pair, zero_cos], dim=-1)

        diff = (a - e).abs()
        prod = a * e
        cos = F.cosine_similarity(a, e, dim=-1, eps=1e-8).unsqueeze(-1)
        return torch.cat([a, e, diff, prod, cos], dim=-1)

    @staticmethod
    def output_dim(hidden_dim: int) -> int:
        return 4 * hidden_dim + 1
