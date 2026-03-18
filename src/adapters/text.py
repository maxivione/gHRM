from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TextAdapterConfig:
    input_dim: int
    workspace_dim: int


class TextAdapter(nn.Module):
    def __init__(self, config: TextAdapterConfig) -> None:
        super().__init__()
        self.projection = nn.Linear(config.input_dim, config.workspace_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(inputs)
