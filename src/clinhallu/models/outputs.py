"""Typed model outputs shared by training and inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class HKGFusionOutput:
    logits: torch.Tensor = field(default=None)
    probs: torch.Tensor = field(default=None)
    p_hallucinated: torch.Tensor = field(default=None)
    loss: Optional[torch.Tensor] = None
    c: torch.Tensor = field(default=None)
    e: torch.Tensor = field(default=None)


@dataclass
class GAERPlusPlusOutput:
    logits: torch.Tensor = field(default=None)
    probs: torch.Tensor = field(default=None)
    p_hallucinated: torch.Tensor = field(default=None)
    loss: Optional[torch.Tensor] = None
    c: torch.Tensor = field(default=None)
    a: torch.Tensor = field(default=None)
    e: torch.Tensor = field(default=None)
    grounding: torch.Tensor = field(default=None)
    evidence_stats: Optional[dict] = None
    feature_dim: int = 0
