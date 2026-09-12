"""Answer-token pooling used by GAER++."""

from __future__ import annotations

import torch
import torch.nn as nn


class AnswerAttentionPool(nn.Module):
    """Learn a masked attention distribution over answer tokens."""

    def __init__(self, hidden_dim: int = 768, attn_hidden_dim: int = 256) -> None:
        super().__init__()
        # Attribute names intentionally match the original implementation so
        # existing GAER++ state dictionaries load without key translation.
        self.proj = nn.Linear(hidden_dim, attn_hidden_dim)
        self.score = nn.Linear(attn_hidden_dim, 1, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        answer_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = answer_mask.bool() & attention_mask.bool()
        scores = self.score(torch.tanh(self.proj(hidden))).squeeze(-1)
        fill = torch.finfo(scores.dtype).min
        scores = torch.where(valid, scores, torch.full_like(scores, fill))
        weights = torch.softmax(scores, dim=1) * valid.to(hidden.dtype)
        pooled = torch.einsum("bn,bnd->bd", weights, hidden)
        has_answer = valid.any(dim=1, keepdim=True)
        return torch.where(has_answer, pooled, torch.zeros_like(pooled))


class AnswerRepresentation(nn.Module):
    """Compute ``a = Linear([H_CLS ; AttentionPool(H_A)])``."""

    def __init__(self, hidden_dim: int = 768, attn_hidden_dim: int = 256) -> None:
        super().__init__()
        self.attention_pool = AnswerAttentionPool(hidden_dim, attn_hidden_dim)
        self.fuse = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        cls: torch.Tensor,
        answer_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        answer = self.attention_pool(hidden, answer_mask, attention_mask)
        return self.fuse(torch.cat([cls, answer], dim=-1))
