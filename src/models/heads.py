from __future__ import annotations

import torch
from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class ClassifierHeadConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int
    dropout: float


class ClassifierHead(nn.Module):
    def __init__(self, config: ClassifierHeadConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(config.input_dim),
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    def forward(self, hidden_state):
        return self.layers(hidden_state)
