"""Legacy context-aware attention retained for checkpoint compatibility."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class ContextAwareAttention(nn.Module):
    def __init__(self, hidden_dim: int = 768, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        H: torch.Tensor,
        context_mask: torch.Tensor,
        answer_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        context_mask = context_mask.bool()
        answer_mask = answer_mask.bool()
        real = attention_mask.bool()

        allowed_key = context_mask & real
        key_padding_mask = ~allowed_key

        empty_context = ~allowed_key.any(dim=1)
        if empty_context.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[empty_context, 0] = False

        attn_out, attn_weights = self.mha(
            query=H,
            key=H,
            value=H,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )

        attended_H = self.layer_norm(H + attn_out)

        answer_mask_f = answer_mask.unsqueeze(-1).to(H.dtype)
        attended_H = attended_H * answer_mask_f + H * (1.0 - answer_mask_f)

        support_score = attn_weights.max(dim=-1).values
        support_score = support_score * answer_mask.to(support_score.dtype)

        return attended_H, support_score
