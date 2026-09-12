"""Classification heads for GAER++ and legacy HKG-Fusion models."""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureConcatenation(nn.Module):
    def forward(self, c: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return torch.cat([c, e], dim=-1)


class HallucinationClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.net(r.to(self.net[0].weight.dtype))


class GAERClassifierHead(nn.Module):
    """GAER++ MLP: input -> 1024 -> 512 -> 128 -> one raw logit."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int, int] = (1024, 512, 128),
        dropouts: tuple[float, float, float] = (0.20, 0.15, 0.10),
    ) -> None:
        super().__init__()
        h1, h2, h3 = hidden_dims
        d1, d2, d3 = dropouts
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.GELU(),
            nn.LayerNorm(h1),
            nn.Dropout(d1),
            nn.Linear(h1, h2),
            nn.GELU(),
            nn.LayerNorm(h2),
            nn.Dropout(d2),
            nn.Linear(h2, h3),
            nn.GELU(),
            nn.Dropout(d3),
            nn.Linear(h3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.to(self.net[0].weight.dtype)).squeeze(-1)
