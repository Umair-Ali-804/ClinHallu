"""Masked answer/context cross-attention for context-derived evidence."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class BidirectionalCrossAttention(nn.Module):
    """Answer/context cross-attention without residual evidence mixing.

    The answer-to-context output is derived only from context values, so the
    downstream evidence vector cannot accidentally pool answer states. Empty
    masks receive one fallback key to prevent attention NaNs.
    """

    def __init__(self, hidden_dim: int = 768, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.answer_to_context = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.context_to_answer = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # LayerNorm applied to the raw attention output only (no residual add of H).
        self.ln_a2c = nn.LayerNorm(hidden_dim)
        self.ln_c2a = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _key_padding_mask(select_mask: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        allowed_key = select_mask.bool() & real
        key_padding_mask = ~allowed_key
        empty_rows = ~allowed_key.any(dim=1)
        if empty_rows.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[empty_rows, 0] = False
        return key_padding_mask

    def forward(
        self,
        H: torch.Tensor,
        context_mask: torch.Tensor,
        answer_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        compute_reverse: bool = True,
        need_weights: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one or both directions and optionally return attention weights.

        When ``compute_reverse`` is false, the reverse output and its weights
        are returned as ``None``.
        """
        real = attention_mask.bool()
        context_mask = context_mask.bool()
        answer_mask = answer_mask.bool()

        context_key_mask = self._key_padding_mask(context_mask, real)

        z_raw, a2c_weights = self.answer_to_context(
            query=H,
            key=H,
            value=H,
            key_padding_mask=context_key_mask,
            need_weights=need_weights,
            average_attn_weights=True,
        )
        z = self.ln_a2c(z_raw)

        if not compute_reverse:
            return z, None, a2c_weights, None

        answer_key_mask = self._key_padding_mask(answer_mask, real)
        a_raw, c2a_weights = self.context_to_answer(
            query=H,
            key=H,
            value=H,
            key_padding_mask=answer_key_mask,
            need_weights=need_weights,
            average_attn_weights=True,
        )
        answer_ctx = self.ln_c2a(a_raw)

        return z, answer_ctx, a2c_weights, c2a_weights
